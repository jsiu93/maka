/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import type { RuntimeExecutionConnection } from '@maka/core/llm-connections';
import { lookupModelMetadata } from '@maka/core/model-metadata';
import { relayModelProfile } from '@maka/core/model-thinking';
import type { ContextBudgetPolicy } from './context-budget.js';
import { finitePositive } from './context-budget-helpers.js';

export interface BuildDefaultContextBudgetPolicyOptions {
  name?: string;
  env?: Record<string, string | undefined>;
  modelId?: string;
}

export function buildDefaultContextBudgetPolicy(
  connection: RuntimeExecutionConnection,
  options: BuildDefaultContextBudgetPolicyOptions = {},
): ContextBudgetPolicy | undefined {
  const env = options.env ?? process.env;
  if (env.MAKA_CONTEXT_BUDGET === 'off') return undefined;
  const contextWindow = resolveSelectedModelContextWindow(connection, options.modelId);
  const reserveTokens = defaultCompactReserveTokens(env, contextWindow);
  const maxHistoryEstimatedTokens = defaultHistoryBudgetTokens(
    connection,
    contextWindow,
    reserveTokens,
  );
  const surfaceName = (options.name ?? 'default-history-budget').replace(
    /-default-history-budget$/,
    '',
  );
  const staleToolResultPrune = buildStaleToolResultPrunePolicy(env);
  const historyCompact = buildHistoryCompactPolicy(
    env,
    `${surfaceName}-history-compact`,
    reserveTokens,
  );
  const semanticCompact = buildSemanticCompactPolicy(env, `${surfaceName}-semantic-compact`);
  const activeToolResultPrune = buildActiveToolResultPrunePolicy(env);
  if (
    maxHistoryEstimatedTokens === undefined &&
    staleToolResultPrune === undefined &&
    historyCompact === undefined &&
    semanticCompact === undefined &&
    activeToolResultPrune === undefined
  ) {
    return undefined;
  }
  return {
    name: options.name ?? 'default-history-budget',
    ...(maxHistoryEstimatedTokens !== undefined ? { maxHistoryEstimatedTokens } : {}),
    ...(staleToolResultPrune !== undefined ? { staleToolResultPrune } : {}),
    ...(historyCompact !== undefined ? { historyCompact } : {}),
    ...(semanticCompact !== undefined ? { semanticCompact } : {}),
    ...(activeToolResultPrune !== undefined ? { activeToolResultPrune } : {}),
  };
}

function buildStaleToolResultPrunePolicy(
  env: Record<string, string | undefined>,
): NonNullable<ContextBudgetPolicy['staleToolResultPrune']> | undefined {
  const enabled = parseOptionalBoolean(
    env.MAKA_CONTEXT_STALE_TOOL_RESULT_PRUNE,
    'MAKA_CONTEXT_STALE_TOOL_RESULT_PRUNE',
  );
  if (enabled === false) return undefined;
  return {
    enabled: true,
    maxResultEstimatedTokens: parsePositiveInt(env.MAKA_CONTEXT_STALE_TOOL_RESULT_MAX_TOKENS, 2048),
    minRecentTurnsFull: parsePositiveInt(env.MAKA_CONTEXT_STALE_TOOL_RESULT_MIN_RECENT_TURNS, 2),
  };
}

function buildActiveToolResultPrunePolicy(
  env: Record<string, string | undefined>,
): NonNullable<ContextBudgetPolicy['activeToolResultPrune']> | undefined {
  const enabled = parseOptionalBoolean(
    env.MAKA_CONTEXT_ACTIVE_TOOL_RESULT_PRUNE,
    'MAKA_CONTEXT_ACTIVE_TOOL_RESULT_PRUNE',
  );
  if (enabled === false) return undefined;
  return {
    enabled: true,
    maxCurrentResultEstimatedTokens: parsePositiveInt(
      env.MAKA_CONTEXT_ACTIVE_TOOL_RESULT_MAX_ESTIMATED_TOKENS,
      2048,
    ),
    minSupersededResultEstimatedTokens: parsePositiveInt(
      env.MAKA_CONTEXT_ACTIVE_TOOL_RESULT_MIN_SUPERSEDED_TOKENS,
      256,
    ),
    minStepNumber:
      parseOptionalNonNegativeInt(env.MAKA_CONTEXT_ACTIVE_TOOL_RESULT_MIN_STEP_NUMBER) ?? 1,
  };
}

function buildHistoryCompactPolicy(
  env: Record<string, string | undefined>,
  defaultHighWaterName: string,
  reserveTokens: number,
): NonNullable<ContextBudgetPolicy['historyCompact']> | undefined {
  const enabled = parseOptionalBoolean(
    env.MAKA_CONTEXT_HISTORY_COMPACT,
    'MAKA_CONTEXT_HISTORY_COMPACT',
  );
  if (enabled === false) return undefined;
  const midTurn = buildHistoryCompactMidTurnPolicy(env, reserveTokens);
  return {
    enabled: true,
    highWaterName: env.MAKA_CONTEXT_HISTORY_COMPACT_HIGH_WATER_NAME ?? defaultHighWaterName,
    ...(midTurn !== undefined ? { midTurn } : {}),
  };
}

// Mid-turn capacity compaction is a runtime-owned default (issue #882 PR 3):
// whenever history compaction is enabled, the runtime derives midTurn from the
// selected model's window (`contextWindow - reserveTokens`, the same reserve as
// the turn-boundary budget) so every surface inherits the invariant without
// copying config. `MAKA_CONTEXT_HISTORY_COMPACT_MID_TURN=off` stays as the
// explicit escape hatch. The backend still gates activation on the checkpoint
// seams, the persisted head anchor, and a KNOWN context window, so a session
// without model metadata (or a child with no anchor seam) never misfires even
// though the default is on.
function buildHistoryCompactMidTurnPolicy(
  env: Record<string, string | undefined>,
  reserveTokens: number,
): NonNullable<NonNullable<ContextBudgetPolicy['historyCompact']>['midTurn']> | undefined {
  const enabled = parseOptionalBoolean(
    env.MAKA_CONTEXT_HISTORY_COMPACT_MID_TURN,
    'MAKA_CONTEXT_HISTORY_COMPACT_MID_TURN',
  );
  if (enabled === false) return undefined;
  const reserveTailEvents = parseOptionalNonNegativeInt(
    env.MAKA_CONTEXT_HISTORY_COMPACT_MID_TURN_TAIL_EVENTS,
  );
  return {
    enabled: true,
    reserveTokens,
    ...(reserveTailEvents !== undefined ? { reserveTailEvents } : {}),
  };
}

// Semantic compaction is the #981/#986 attention-first experiment, distinct
// from the standard historyCompact mechanism. It defaults OFF and is opt-in per
// surface (issue #882 PR 3): an explicit MAKA_CONTEXT_SEMANTIC_COMPACT truthy
// value, or a MAKA_CONTEXT_SEMANTIC_COMPACT_MODE other than `off`, turns it on.
// This keeps the experiment out of every surface's default budget without each
// surface stripping it locally.
function buildSemanticCompactPolicy(
  env: Record<string, string | undefined>,
  defaultHighWaterName: string,
): NonNullable<ContextBudgetPolicy['semanticCompact']> | undefined {
  const enabled = parseOptionalBoolean(
    env.MAKA_CONTEXT_SEMANTIC_COMPACT,
    'MAKA_CONTEXT_SEMANTIC_COMPACT',
  );
  if (enabled === false) return undefined;
  const mode = parseSemanticCompactMode(env.MAKA_CONTEXT_SEMANTIC_COMPACT_MODE);
  if (mode === 'off') return undefined;
  // Default off: require an explicit opt-in (the boolean flag or an explicit
  // non-off mode). Neither present means the experiment stays out of the budget.
  if (enabled !== true && mode === undefined) return undefined;
  const archiveRequired = parseOptionalBoolean(
    env.MAKA_CONTEXT_SEMANTIC_COMPACT_ARCHIVE_REQUIRED,
    'MAKA_CONTEXT_SEMANTIC_COMPACT_ARCHIVE_REQUIRED',
  );
  return {
    enabled: true,
    mode: mode ?? 'replace',
    minStepNumber:
      parseOptionalNonNegativeInt(env.MAKA_CONTEXT_SEMANTIC_COMPACT_MIN_STEP_NUMBER) ?? 2,
    // Attention compaction stays dormant through the first 128K estimated
    // provider tokens. The completed-span thresholds below only apply after
    // this high-water has been crossed.
    highWaterRatio: parseOptionalRatio(env.MAKA_CONTEXT_SEMANTIC_COMPACT_HIGH_WATER_RATIO) ?? 1,
    forceRatio: parseOptionalRatio(env.MAKA_CONTEXT_SEMANTIC_COMPACT_FORCE_RATIO),
    targetRatio: parseOptionalRatio(env.MAKA_CONTEXT_SEMANTIC_COMPACT_TARGET_RATIO),
    maxActiveEstimatedTokens:
      parseOptionalPositiveInt(env.MAKA_CONTEXT_SEMANTIC_COMPACT_MAX_ACTIVE_ESTIMATED_TOKENS) ??
      parseOptionalPositiveInt(env.MAKA_CONTEXT_SEMANTIC_COMPACT_MAX_ESTIMATED_TOKENS) ??
      131_072,
    ...(parseOptionalNonNegativeInt(env.MAKA_CONTEXT_SEMANTIC_COMPACT_MIN_RECENT_MESSAGES) !==
    undefined
      ? {
          minRecentMessages: parseOptionalNonNegativeInt(
            env.MAKA_CONTEXT_SEMANTIC_COMPACT_MIN_RECENT_MESSAGES,
          ),
        }
      : {}),
    ...(parseOptionalNonNegativeInt(env.MAKA_CONTEXT_SEMANTIC_COMPACT_MIN_RECENT_TOOL_PAIRS) !==
    undefined
      ? {
          minRecentToolPairs: parseOptionalNonNegativeInt(
            env.MAKA_CONTEXT_SEMANTIC_COMPACT_MIN_RECENT_TOOL_PAIRS,
          ),
        }
      : {}),
    minSafePrefixEstimatedTokens:
      parseOptionalPositiveInt(
        env.MAKA_CONTEXT_SEMANTIC_COMPACT_MIN_SAFE_PREFIX_ESTIMATED_TOKENS,
      ) ?? 4_096,
    minNewPrefixEstimatedTokens:
      parseOptionalPositiveInt(env.MAKA_CONTEXT_SEMANTIC_COMPACT_MIN_NEW_PREFIX_ESTIMATED_TOKENS) ??
      4_096,
    maxAcceptedProjectionEstimatedTokens:
      parseOptionalPositiveInt(
        env.MAKA_CONTEXT_SEMANTIC_COMPACT_MAX_ACCEPTED_PROJECTION_ESTIMATED_TOKENS,
      ) ??
      parseOptionalPositiveInt(env.MAKA_CONTEXT_SEMANTIC_COMPACT_MAX_SUMMARY_ESTIMATED_TOKENS) ??
      parseOptionalPositiveInt(env.MAKA_CONTEXT_SEMANTIC_COMPACT_SUMMARY_MAX_ESTIMATED_TOKENS) ??
      768,
    ...(parseOptionalPositiveInt(env.MAKA_CONTEXT_SEMANTIC_COMPACT_MAX_SUMMARY_ESTIMATED_TOKENS) !==
    undefined
      ? {
          maxSummaryEstimatedTokens: parseOptionalPositiveInt(
            env.MAKA_CONTEXT_SEMANTIC_COMPACT_MAX_SUMMARY_ESTIMATED_TOKENS,
          ),
        }
      : {}),
    minSavingsTokens:
      parseOptionalNonNegativeInt(env.MAKA_CONTEXT_SEMANTIC_COMPACT_MIN_SAVINGS_TOKENS) ?? 256,
    minSavingsRatio: parseOptionalRatio(env.MAKA_CONTEXT_SEMANTIC_COMPACT_MIN_SAVINGS_RATIO),
    minNetSavingsTokens:
      parseOptionalNonNegativeInt(env.MAKA_CONTEXT_SEMANTIC_COMPACT_MIN_NET_SAVINGS_TOKENS) ?? 256,
    compactCallTokenCostWeight: parseOptionalNonNegativeNumber(
      env.MAKA_CONTEXT_SEMANTIC_COMPACT_CALL_TOKEN_COST_WEIGHT,
    ),
    maxCompactCallTokens:
      parseOptionalPositiveInt(env.MAKA_CONTEXT_SEMANTIC_COMPACT_MAX_CALL_TOKENS) ?? 4096,
    maxConsecutiveInvalidSummaries:
      parseOptionalNonNegativeInt(
        env.MAKA_CONTEXT_SEMANTIC_COMPACT_MAX_CONSECUTIVE_INVALID_SUMMARIES,
      ) ?? 2,
    invalidSummaryCooldownSteps:
      parseOptionalNonNegativeInt(
        env.MAKA_CONTEXT_SEMANTIC_COMPACT_INVALID_SUMMARY_COOLDOWN_STEPS,
      ) ?? 8,
    timeoutMs: parseOptionalPositiveInt(env.MAKA_CONTEXT_SEMANTIC_COMPACT_TIMEOUT_MS),
    ...(archiveRequired !== undefined ? { archiveRequired } : {}),
    ...(env.MAKA_CONTEXT_SEMANTIC_COMPACT_MODEL
      ? { summarizerModel: env.MAKA_CONTEXT_SEMANTIC_COMPACT_MODEL }
      : {}),
    ...(env.MAKA_CONTEXT_SEMANTIC_COMPACT_PROMPT_VERSION
      ? { promptVersion: env.MAKA_CONTEXT_SEMANTIC_COMPACT_PROMPT_VERSION }
      : {}),
    highWaterName: env.MAKA_CONTEXT_SEMANTIC_COMPACT_HIGH_WATER_NAME ?? defaultHighWaterName,
  };
}

// Single owner of the compaction reserve default. The classic 16384 reserve
// assumed large-window models; on an 8K window it derived a 1-token history
// budget and a 1-token mid_turn high water — every multi-step turn ran the
// summarizer for a checkpoint the replay gate could never admit. The default
// is therefore bounded by the KNOWN window (a quarter of it, capped at 16384;
// peers bound the same way: opencode caps its buffer by the model's output
// limit, gemini-cli triggers at a window fraction). An explicit
// MAKA_CONTEXT_HISTORY_COMPACT_RESERVE_TOKENS is respected verbatim, and an
// unknown window keeps the classic constant.
function defaultCompactReserveTokens(
  env: Record<string, string | undefined>,
  contextWindow: number | undefined,
): number {
  const explicit = parseOptionalPositiveInt(env.MAKA_CONTEXT_HISTORY_COMPACT_RESERVE_TOKENS);
  if (explicit !== undefined) return explicit;
  if (contextWindow === undefined) return 16_384;
  return Math.min(16_384, Math.max(1, Math.floor(contextWindow / 4)));
}

function defaultHistoryBudgetTokens(
  connection: RuntimeExecutionConnection,
  contextWindow: number | undefined,
  reserveTokens: number,
): number | undefined {
  if (contextWindow !== undefined) {
    return Math.max(1, contextWindow - reserveTokens);
  }
  if (connection.providerType === 'deepseek') return undefined;
  return 32_000;
}

export function resolveSelectedModelContextWindow(
  connection: RuntimeExecutionConnection,
  modelId: string | undefined,
): number | undefined {
  const selectedModelId = modelId ?? connection.defaultModel;
  if (selectedModelId === undefined) return undefined;
  // A user declaration outranks both the provider's /models report and
  // generated metadata — mirrors the declared-vision precedence in
  // model-metadata.ts. A declared context window is legal on any provider: it
  // states a fact about the model, not a request shape (#1584).
  const declared = relayModelProfile(connection, selectedModelId)?.contextWindow;
  if (declared !== undefined) return declared;
  const model = connection.models?.find((candidate) => candidate.id === selectedModelId);
  const metadata = lookupModelMetadata(connection.providerType, selectedModelId);
  // Provider/access-path facts outrank static metadata. Within one source,
  // use the narrowest positive bound: models.dev's input limit can be lower
  // than its total context window, while an access path can expose a narrower
  // context window than the public catalog.
  const modelLimit = narrowestPositiveLimit(model?.contextWindow, model?.inputLimit);
  const metadataLimit = narrowestPositiveLimit(metadata.contextWindow, metadata.inputLimit);
  return modelLimit ?? metadataLimit;
}

function narrowestPositiveLimit(...values: Array<number | undefined>): number | undefined {
  const positiveValues = values.filter(
    (value): value is number => typeof value === 'number' && Number.isFinite(value) && value > 0,
  );
  return positiveValues.length > 0 ? Math.min(...positiveValues) : undefined;
}

export interface ContextBudgetCapacity {
  tokens: number;
  source: 'selected_model' | 'policy_fallback';
}

export function resolveContextBudgetCapacity(
  connection: RuntimeExecutionConnection,
  modelId: string | undefined,
  policy: ContextBudgetPolicy | undefined,
): ContextBudgetCapacity | undefined {
  const selectedWindow = resolveSelectedModelContextWindow(connection, modelId);
  if (selectedWindow !== undefined) {
    return { tokens: selectedWindow, source: 'selected_model' };
  }

  const historyBudget = finitePositive(policy?.maxHistoryEstimatedTokens);
  const reserveTokens = finitePositive(policy?.historyCompact?.midTurn?.reserveTokens);
  if (historyBudget === undefined || reserveTokens === undefined) return undefined;
  return { tokens: historyBudget + reserveTokens, source: 'policy_fallback' };
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  const parsed = parseOptionalPositiveInt(value);
  return parsed ?? fallback;
}

function parseOptionalPositiveInt(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

function parseOptionalNonNegativeInt(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : undefined;
}

function parseOptionalNonNegativeNumber(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : undefined;
}

function parseOptionalRatio(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.min(1, parsed) : undefined;
}

function parseSemanticCompactMode(
  value: string | undefined,
): NonNullable<ContextBudgetPolicy['semanticCompact']>['mode'] | undefined {
  if (!value) return undefined;
  if (
    value === 'off' ||
    value === 'validate_only' ||
    value === 'prepare_step_dry_run' ||
    value === 'replace'
  )
    return value;
  return undefined;
}

function parseOptionalBoolean(value: string | undefined, name: string): boolean | undefined {
  const normalized = value?.trim().toLowerCase();
  if (!normalized) return undefined;
  switch (normalized) {
    case '1':
    case 'true':
    case 'yes':
    case 'on':
    case 'enabled':
      return true;
    case '0':
    case 'false':
    case 'no':
    case 'off':
    case 'disabled':
      return false;
    default:
      throw new Error(`${name} must be a boolean, got ${JSON.stringify(value)}`);
  }
}
