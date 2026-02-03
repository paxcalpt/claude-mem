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

  constructor(dbManager: DatabaseManager, sessionManager: SessionManager) {
    this.dbManager = dbManager;
    this.sessionManager = sessionManager;
  }

  /**
   * Set the fallback agent (Claude SDK) for when Gemini CLI fails
   * Must be set after construction to avoid circular dependency
   */
  setFallbackAgent(agent: FallbackAgent): void {
    this.fallbackAgent = agent;
  }

  /**
   * Start Gemini CLI agent for a session
   * Uses multi-turn conversation to maintain context across messages
   */
  async startSession(session: ActiveSession, worker?: WorkerRef): Promise<void> {
    try {
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
      const initResponse = await this.queryGeminiCliWithRetry(session.conversationHistory, cliPath, model);

      if (initResponse.content) {
        // Add response to conversation history
        session.conversationHistory.push({ role: 'assistant', content: initResponse.content });

        // Track token usage (Gemini CLI provides token counts)
        const tokensUsed = initResponse.tokensUsed || 0;
        session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);  // Rough estimate
        session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);

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
          const obsResponse = await this.queryGeminiCliWithRetry(session.conversationHistory, cliPath, model);

          let tokensUsed = 0;
          if (obsResponse.content) {
            // Add response to conversation history
            session.conversationHistory.push({ role: 'assistant', content: obsResponse.content });

            tokensUsed = obsResponse.tokensUsed || 0;
            session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
            session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
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
          const summaryResponse = await this.queryGeminiCliWithRetry(session.conversationHistory, cliPath, model);

          let tokensUsed = 0;
          if (summaryResponse.content) {
            // Add response to conversation history
            session.conversationHistory.push({ role: 'assistant', content: summaryResponse.content });

            tokensUsed = summaryResponse.tokensUsed || 0;
            session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
            session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
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
        historyLength: session.conversationHistory.length
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
   * Query Gemini CLI with retry logic for rate limits
   */
  private async queryGeminiCliWithRetry(
    history: { role: 'user' | 'assistant'; content: string }[],
    cliPath: string,
    model: string,
    maxRetries: number = 3
  ): Promise<{ content: string; tokensUsed?: number }> {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        return await this.queryGeminiCli(history, cliPath, model);
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
    model: string
  ): Promise<{ content: string; tokensUsed?: number }> {
    const totalChars = history.reduce((sum, m) => sum + m.content.length, 0);

    logger.debug('SDK', `Querying Gemini CLI (${model})`, {
      turns: history.length,
      totalChars
    });

    // Build conversation format for prompt
    // Include full history for context
    const conversationText = history
      .map(msg => {
        const role = msg.role === 'assistant' ? 'Assistant' : 'User';
        return `${role}: ${msg.content}`;
      })
      .join('\n\n');

    return new Promise((resolve, reject) => {
      // Use -p flag for inline prompt with JSON output format
      const args = [
        '-p', conversationText,
        '--output-format', 'json',
        '-m', model
      ];

      const child = spawn(cliPath, args, {
        stdio: ['pipe', 'pipe', 'pipe']
      });

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
        reject(new Error(`Gemini CLI error: ${error.message}`));
      });

      child.on('close', (code) => {
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

          // Calculate token usage from stats if available
          let tokensUsed: number | undefined;
          if (jsonResponse.stats) {
            const inputTokens = jsonResponse.stats.input_tokens || 0;
            const outputTokens = jsonResponse.stats.output_tokens || 0;
            tokensUsed = inputTokens + outputTokens;
          }

          resolve({ content, tokensUsed });
        } catch (error) {
          logger.error('SDK', 'Failed to parse Gemini CLI JSON', { stdout: stdout.substring(0, 200) });
          reject(new Error(`Failed to parse Gemini CLI JSON: ${error instanceof Error ? error.message : String(error)}`));
        }
      });

      // Close stdin (no input needed with -p flag)
      child.stdin.end();
    });
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
