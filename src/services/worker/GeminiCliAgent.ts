/**
 * GeminiCliAgent: Gemini CLI-based observation extraction
 *
 * Alternative to GeminiAgent that uses Google's Gemini CLI tool
 * for extracting observations from tool usage.
 *
 * Responsibility:
 * - Call Gemini CLI for observation extraction
 * - Parse responses (same format as Claude)
 * - Sync to database and Chroma
 */

import path from 'path';
import { homedir } from 'os';
import { spawn } from 'child_process';
import { DatabaseManager } from './DatabaseManager.js';
import { SessionManager } from './SessionManager.js';
import { logger } from '../../utils/logger.js';
import { buildInitPrompt, buildObservationPrompt, buildSummaryPrompt, buildContinuationPrompt } from '../../sdk/prompts.js';
import { SettingsDefaultsManager } from '../../shared/SettingsDefaultsManager.js';
import type { ActiveSession } from '../worker-types.js';
import { ModeManager } from '../domain/ModeManager.js';
import { registerProcess, unregisterProcess } from './ProcessRegistry.js';
import {
  processAgentResponse,
  shouldFallbackToClaude,
  isAbortError,
  type WorkerRef,
  type FallbackAgent
} from './agents/index.js';

export class GeminiCliAgent {
  private dbManager: DatabaseManager;
  private sessionManager: SessionManager;
  private fallbackAgent: FallbackAgent | null = null;
  // Track Gemini CLI session UUIDs for session reuse
  private geminiSessionIds: Map<string, string> = new Map(); // memorySessionId -> gemini session_id
  // Concurrency control to prevent resource exhaustion
  private concurrencyLimit: number;
  private activeProcesses = 0;
  private queuedRequests: Array<() => void> = [];

  constructor(dbManager: DatabaseManager, sessionManager: SessionManager) {
    this.dbManager = dbManager;
    this.sessionManager = sessionManager;

    // Load concurrency limit from settings
    const settingsPath = path.join(homedir(), '.claude-mem', 'settings.json');
    const settings = SettingsDefaultsManager.loadFromFile(settingsPath);
    this.concurrencyLimit = parseInt(settings.CLAUDE_MEM_GEMINI_CLI_MAX_CONCURRENT_PROCESSES, 10) || 10;
  }

  /**
   * Set the fallback agent (Claude SDK) for when Gemini CLI fails
   * Must be set after construction to avoid circular dependency
   */
  setFallbackAgent(agent: FallbackAgent): void {
    this.fallbackAgent = agent;
  }

  /**
   * Acquire a process slot before spawning Gemini CLI
   * Implements concurrency limiting to prevent resource exhaustion
   */
  private async acquireProcessSlot(): Promise<void> {
    if (this.activeProcesses < this.concurrencyLimit) {
      this.activeProcesses++;
      return;
    }

    // Wait for slot to become available
    return new Promise(resolve => {
      this.queuedRequests.push(resolve);
    });
  }

  /**
   * Release a process slot after Gemini CLI completes
   * Processes next queued request if any
   */
  private releaseProcessSlot(): void {
    this.activeProcesses--;

    // Process next queued request
    const next = this.queuedRequests.shift();
    if (next) {
      this.activeProcesses++;
      next();
    }
  }

  /**
   * Start Gemini CLI agent for a session
   * Uses multi-turn conversation to maintain context across messages
   */
  async startSession(session: ActiveSession, worker?: WorkerRef): Promise<void> {
    try {
      // Capture/generate memory session ID if not yet set
      // Unlike SDKAgent which gets session_id from Claude SDK, Gemini CLI generates its own
      if (!session.memorySessionId) {
        // Use crypto.randomUUID() to generate a unique session ID
        const { randomUUID } = await import('crypto');
        session.memorySessionId = randomUUID();

        // Persist to database for cross-restart recovery
        this.dbManager.getSessionStore().updateMemorySessionId(
          session.sessionDbId,
          session.memorySessionId
        );

        // Verify the update by reading back from DB
        const verification = this.dbManager.getSessionStore().getSessionById(session.sessionDbId);
        const dbVerified = verification?.memory_session_id === session.memorySessionId;
        logger.info('SESSION', `MEMORY_ID_GENERATED | sessionDbId=${session.sessionDbId} | memorySessionId=${session.memorySessionId} | dbVerified=${dbVerified}`, {
          sessionId: session.sessionDbId,
          memorySessionId: session.memorySessionId
        });

        if (!dbVerified) {
          logger.error('SESSION', `MEMORY_ID_MISMATCH | sessionDbId=${session.sessionDbId} | expected=${session.memorySessionId} | got=${verification?.memory_session_id}`, {
            sessionId: session.sessionDbId
          });
        }
      }

      // Get Gemini CLI configuration
      const { cliPath, model } = this.getGeminiCliConfig();

      // Load active mode
      const mode = ModeManager.getInstance().getActiveMode();

      // Build initial prompt
      const initPrompt = session.lastPromptNumber === 1
        ? buildInitPrompt(session.project, session.contentSessionId, session.userPrompt, mode)
        : buildContinuationPrompt(session.userPrompt, session.lastPromptNumber, session.contentSessionId, mode);

      // Add to conversation history and query Gemini CLI with full context
      session.conversationHistory.push({ role: 'user', content: initPrompt });
      const initResponse = await this.queryGeminiCliWithRetry(session.conversationHistory, cliPath, model, session.sessionDbId);

      if (initResponse.content) {
        // Add response to conversation history
        session.conversationHistory.push({ role: 'assistant', content: initResponse.content });

        // Track token usage (Gemini CLI provides accurate token counts)
        const tokensUsed = initResponse.tokensUsed || 0;
        session.cumulativeInputTokens += initResponse.inputTokens || 0;
        session.cumulativeOutputTokens += initResponse.outputTokens || 0;

        // Process response using shared ResponseProcessor (no original timestamp for init - not from queue)
        await processAgentResponse(
          initResponse.content,
          session,
          this.dbManager,
          this.sessionManager,
          worker,
          tokensUsed,
          null,
          'Gemini CLI'
        );
      } else {
        logger.error('SDK', 'Empty Gemini CLI init response - session may lack context', {
          sessionId: session.sessionDbId,
          model
        });
      }

      // Process pending messages
      // Track cwd from messages for CLAUDE.md generation
      let lastCwd: string | undefined;

      for await (const message of this.sessionManager.getMessageIterator(session.sessionDbId)) {
        // Capture cwd from each message for worktree support
        if (message.cwd) {
          lastCwd = message.cwd;
        }
        // Capture earliest timestamp BEFORE processing (will be cleared after)
        // This ensures backlog messages get their original timestamps, not current time
        const originalTimestamp = session.earliestPendingTimestamp;

        if (message.type === 'observation') {
          // Update last prompt number
          if (message.prompt_number !== undefined) {
            session.lastPromptNumber = message.prompt_number;
          }

          // Build observation prompt
          const obsPrompt = buildObservationPrompt({
            id: 0,
            tool_name: message.tool_name!,
            tool_input: JSON.stringify(message.tool_input),
            tool_output: JSON.stringify(message.tool_response),
            created_at_epoch: originalTimestamp ?? Date.now(),
            cwd: message.cwd
          });

          // Add to conversation history and query Gemini CLI with full context
          session.conversationHistory.push({ role: 'user', content: obsPrompt });
          const obsResponse = await this.queryGeminiCliWithRetry(session.conversationHistory, cliPath, model, session.sessionDbId);

          let tokensUsed = 0;
          if (obsResponse.content) {
            // Add response to conversation history
            session.conversationHistory.push({ role: 'assistant', content: obsResponse.content });

            tokensUsed = obsResponse.tokensUsed || 0;
            session.cumulativeInputTokens += obsResponse.inputTokens || 0;
            session.cumulativeOutputTokens += obsResponse.outputTokens || 0;
          }

          // Process response using shared ResponseProcessor
          await processAgentResponse(
            obsResponse.content || '',
            session,
            this.dbManager,
            this.sessionManager,
            worker,
            tokensUsed,
            originalTimestamp,
            'Gemini CLI',
            lastCwd
          );

        } else if (message.type === 'summarize') {
          // Build summary prompt
          const summaryPrompt = buildSummaryPrompt({
            id: session.sessionDbId,
            memory_session_id: session.memorySessionId,
            project: session.project,
            user_prompt: session.userPrompt,
            last_assistant_message: message.last_assistant_message || ''
          }, mode);

          // Add to conversation history and query Gemini CLI with full context
          session.conversationHistory.push({ role: 'user', content: summaryPrompt });
          const summaryResponse = await this.queryGeminiCliWithRetry(session.conversationHistory, cliPath, model, session.sessionDbId);

          let tokensUsed = 0;
          if (summaryResponse.content) {
            // Add response to conversation history
            session.conversationHistory.push({ role: 'assistant', content: summaryResponse.content });

            tokensUsed = summaryResponse.tokensUsed || 0;
            session.cumulativeInputTokens += summaryResponse.inputTokens || 0;
            session.cumulativeOutputTokens += summaryResponse.outputTokens || 0;
          }

          // Process response using shared ResponseProcessor
          await processAgentResponse(
            summaryResponse.content || '',
            session,
            this.dbManager,
            this.sessionManager,
            worker,
            tokensUsed,
            originalTimestamp,
            'Gemini CLI',
            lastCwd
          );
        }
      }

      // Mark session complete
      const sessionDuration = Date.now() - session.startTime;
      logger.success('SDK', 'Gemini CLI agent completed', {
        sessionId: session.sessionDbId,
        duration: `${(sessionDuration / 1000).toFixed(1)}s`,
        historyLength: session.conversationHistory.length,
        inputTokens: session.cumulativeInputTokens,
        outputTokens: session.cumulativeOutputTokens,
        totalTokens: session.cumulativeInputTokens + session.cumulativeOutputTokens
      });

      // Log final concurrency stats
      logger.info('SDK', 'Gemini CLI concurrency stats at session end', {
        sessionId: session.sessionDbId,
        activeProcesses: this.activeProcesses,
        queuedRequests: this.queuedRequests.length,
        concurrencyLimit: this.concurrencyLimit
      });

    } catch (error: unknown) {
      if (isAbortError(error)) {
        logger.warn('SDK', 'Gemini CLI agent aborted', { sessionId: session.sessionDbId });
        throw error;
      }

      // Check if we should fall back to Claude
      if (shouldFallbackToClaude(error) && this.fallbackAgent) {
        logger.warn('SDK', 'Gemini CLI failed, falling back to Claude SDK', {
          sessionDbId: session.sessionDbId,
          error: error instanceof Error ? error.message : String(error),
          historyLength: session.conversationHistory.length
        });

        // Fall back to Claude - it will use the same session with shared conversationHistory
        // Note: With claim-and-delete queue pattern, messages are already deleted on claim
        return this.fallbackAgent.startSession(session, worker);
      }

      logger.failure('SDK', 'Gemini CLI agent error', { sessionDbId: session.sessionDbId }, error as Error);
      throw error;
    }
  }

  /**
   * Truncate conversation history to prevent CLI argument overflow
   * Implements sliding window: keeps most recent messages within size limits
   */
  private truncateHistory(
    history: { role: 'user' | 'assistant'; content: string }[]
  ): { role: 'user' | 'assistant'; content: string }[] {
    const MAX_MESSAGES = 20;  // Configurable via settings in future
    const MAX_CHARS = 50000;   // Prevent CLI arg overflow (OS limit ~128KB)

    // Fast path: no truncation needed
    if (history.length <= MAX_MESSAGES) {
      const totalChars = history.reduce((sum, m) => sum + m.content.length, 0);
      if (totalChars <= MAX_CHARS) {
        return history;
      }
    }

    // Sliding window: keep most recent messages
    const truncated: typeof history = [];
    let charCount = 0;

    for (let i = history.length - 1; i >= 0; i--) {
      const msg = history[i];
      const msgChars = msg.content.length;

      if (truncated.length >= MAX_MESSAGES || charCount + msgChars > MAX_CHARS) {
        logger.warn('SDK', 'GeminiCliAgent context truncated', {
          originalMessages: history.length,
          keptMessages: truncated.length,
          originalChars: history.reduce((s, m) => s + m.content.length, 0),
          keptChars: charCount
        });
        break;
      }

      truncated.unshift(msg);
      charCount += msgChars;
    }

    return truncated;
  }

  /**
   * Query Gemini CLI with retry logic for rate limits
   */
  private async queryGeminiCliWithRetry(
    history: { role: 'user' | 'assistant'; content: string }[],
    cliPath: string,
    model: string,
    sessionDbId: number,
    maxRetries: number = 3
  ): Promise<{ content: string; tokensUsed?: number; inputTokens?: number; outputTokens?: number }> {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        return await this.queryGeminiCli(history, cliPath, model, sessionDbId);
      } catch (error) {
        const isRateLimit = error instanceof Error && error.message.includes('RATE_LIMIT');

        if (isRateLimit && attempt < maxRetries - 1) {
          const waitTime = Math.pow(2, attempt) * 1000; // Exponential backoff: 1s, 2s, 4s
          logger.warn('SDK', `Rate limited, waiting ${waitTime}ms before retry ${attempt + 1}/${maxRetries}`, { model });
          await new Promise(resolve => setTimeout(resolve, waitTime));
          continue;
        }
        throw error;
      }
    }
    throw new Error('Failed after max retries');
  }

  /**
   * Query Gemini CLI with full conversation history (multi-turn)
   * Sends the entire conversation context for coherent responses
   * Uses -p flag with --output-format json for structured output
   */
  private async queryGeminiCli(
    history: { role: 'user' | 'assistant'; content: string }[],
    cliPath: string,
    model: string,
    sessionDbId: number
  ): Promise<{ content: string; tokensUsed?: number; inputTokens?: number; outputTokens?: number }> {
    // Acquire process slot to enforce concurrency limit
    await this.acquireProcessSlot();

    const totalChars = history.reduce((sum, m) => sum + m.content.length, 0);

    logger.debug('SDK', `Querying Gemini CLI (${model})`, {
      turns: history.length,
      totalChars,
      activeProcesses: this.activeProcesses,
      queuedRequests: this.queuedRequests.length
    });

    // Truncate history to prevent CLI argument overflow
    const truncatedHistory = this.truncateHistory(history);

    // Build conversation format for prompt
    // Include full history for context
    const conversationText = truncatedHistory
      .map(msg => {
        const role = msg.role === 'assistant' ? 'Assistant' : 'User';
        return `${role}: ${msg.content}`;
      })
      .join('\n\n');

    try {
      return await new Promise((resolve, reject) => {
        // Use -p flag for inline prompt with JSON output format
        const args = [
          '-p', conversationText,
          '--output-format', 'json',
          '-m', model
        ];

        // TODO: Session resumption optimization
        // The Gemini CLI supports --resume flag to continue previous sessions,
        // which would enable automatic context caching and reduce token costs by 50-90%.
        // Implementation requires:
        // 1. Store session_id from first response in geminiSessionIds map
        // 2. Use --resume flag with stored session on subsequent queries
        // 3. Handle session expiration/cleanup
        // However, --resume works with session index numbers (not UUIDs) which change
        // as new sessions are created, making reliable session tracking complex.
        // For now, rely on Gemini's implicit caching of similar prompts.

        const child = spawn(cliPath, args, {
          stdio: ['pipe', 'pipe', 'pipe']
        });

        // Register process in ProcessRegistry for proper cleanup tracking
        if (child.pid) {
          registerProcess(child.pid, sessionDbId, child);
        }

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => {
          stdout += data.toString();
        });

        child.stderr.on('data', (data) => {
          stderr += data.toString();
        });

        child.on('error', (error) => {
          logger.error('SDK', 'Gemini CLI process error', {}, error);
          // Unregister on error
          if (child.pid) {
            unregisterProcess(child.pid);
          }
          reject(new Error(`Gemini CLI error: ${error.message}`));
        });

        child.on('close', (code) => {
          // Unregister process on close
          if (child.pid) {
            unregisterProcess(child.pid);
          }

          // Check for rate limit errors in stdout/stderr
          const output = stdout + stderr;
          if (output.includes('exhausted your capacity') || output.includes('quota') || output.includes('rate limit')) {
            logger.warn('SDK', 'Gemini CLI rate limited, will retry', { model });
            reject(new Error('RATE_LIMIT: Gemini CLI rate limited'));
            return;
          }

        // Check if stderr only contains deprecation warnings (not real errors)
        const hasOnlyWarnings = stderr.trim() &&
          stderr.includes('DeprecationWarning') &&
          !stderr.toLowerCase().includes('error:') &&
          !stderr.toLowerCase().includes('failed');

        // Treat exit code 1 with only warnings as success if we have stdout
        const isSuccess = code === 0 || (code === 1 && hasOnlyWarnings && stdout.trim());

        if (!isSuccess) {
          logger.error('SDK', 'Gemini CLI exited with error', { code, stderr });
          reject(new Error(`Gemini CLI exited with code ${code}: ${stderr}`));
          return;
        }

        if (!stdout.trim()) {
          logger.error('SDK', 'Empty response from Gemini CLI');
          resolve({ content: '' });
          return;
        }

        try {
          // Parse JSON response
          const jsonResponse = JSON.parse(stdout);

          // Extract response text and token stats
          const content = jsonResponse.response || '';

          // Store Gemini session_id for potential future session reuse
          const geminiSessionId = jsonResponse.session_id;
          if (geminiSessionId) {
            logger.debug('SDK', 'Gemini CLI session created', { sessionId: geminiSessionId });
            // Note: Currently not using session resumption due to complexity
            // of tracking session indices vs UUIDs. Future enhancement.
          }

          // Calculate token usage from stats if available
          let tokensUsed: number | undefined;
          let inputTokens = 0;
          let outputTokens = 0;

          if (jsonResponse.stats) {
            inputTokens = jsonResponse.stats.input_tokens ?? 0;
            outputTokens = jsonResponse.stats.output_tokens ?? 0;
            tokensUsed = inputTokens + outputTokens;

            // Track cache hits if available (for monitoring optimization impact)
            const cacheHits = jsonResponse.stats.cache_hits ?? 0;
            const cachedTokens = jsonResponse.stats.cached_tokens ?? 0;

            if (cacheHits > 0) {
              const savingsPercent = tokensUsed > 0 ? Math.round((cachedTokens / tokensUsed) * 100) : 0;
              logger.info('SDK', 'Gemini CLI cache hit', {
                cacheHits,
                cachedTokens,
                totalTokens: tokensUsed,
                savings: `${savingsPercent}%`
              });
            }
          }

          resolve({ content, tokensUsed, inputTokens, outputTokens });
        } catch (error) {
          logger.error('SDK', 'Failed to parse Gemini CLI JSON', { stdout: stdout.substring(0, 200) });
          reject(new Error(`Failed to parse Gemini CLI JSON: ${error instanceof Error ? error.message : String(error)}`));
        }
      });

        // Close stdin (no input needed with -p flag)
        child.stdin.end();
      });
    } finally {
      // Always release process slot, even if spawn fails
      this.releaseProcessSlot();
    }
  }

  /**
   * Get Gemini CLI configuration from settings
   */
  private getGeminiCliConfig(): { cliPath: string; model: string } {
    const settingsPath = path.join(homedir(), '.claude-mem', 'settings.json');
    const settings = SettingsDefaultsManager.loadFromFile(settingsPath);

    // CLI path: from settings or auto-detect via PATH
    let cliPath = settings.CLAUDE_MEM_GEMINI_CLI_PATH || 'gemini';

    // If empty string, use 'gemini' (will search PATH)
    if (!cliPath.trim()) {
      cliPath = 'gemini';
    }

    // Model: from settings or default
    const model = settings.CLAUDE_MEM_GEMINI_CLI_MODEL || 'gemini-2.5-flash';

    return { cliPath, model };
  }
}

/**
 * Check if Gemini CLI is available (can find the binary)
 */
export function isGeminiCliAvailable(): boolean {
  const settingsPath = path.join(homedir(), '.claude-mem', 'settings.json');
  const settings = SettingsDefaultsManager.loadFromFile(settingsPath);

  // For now, assume it's available if the path is set or default 'gemini' command exists
  // A more robust check would be to actually try running `gemini --version`
  return true;
}

/**
 * Check if Gemini CLI is the selected provider
 */
export function isGeminiCliSelected(): boolean {
  const settingsPath = path.join(homedir(), '.claude-mem', 'settings.json');
  const settings = SettingsDefaultsManager.loadFromFile(settingsPath);
  return settings.CLAUDE_MEM_PROVIDER === 'gemini-cli';
}
