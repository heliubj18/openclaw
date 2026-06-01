import fs from "node:fs";
import path from "node:path";
import { colorize, isRich, theme } from "../../packages/terminal-core/src/theme.js";
import { readAcpSessionMetaForEntry } from "../acp/runtime/session-meta.js";
import { getRuntimeConfig } from "../config/config.js";
import { loadSessionStore } from "../config/sessions.js";
import { resolveSessionFilePath } from "../config/sessions/paths.js";
import type { SessionEntry } from "../config/sessions/types.js";
import { isTerminalSessionStatus } from "../config/sessions/types.js";
import { resolveStoredSessionKeyForAgentStore } from "../gateway/session-store-key.js";
import { getDiagnosticSessionActivitySnapshot } from "../logging/diagnostic-run-activity.js";
import { type RuntimeEnv, writeRuntimeJson } from "../runtime.js";
import { isPidAlive } from "../shared/pid-alive.js";
import { resolveTrajectoryFilePath } from "../trajectory/paths.js";
import { resolveTrajectoryRuntimeFile } from "../trajectory/runtime-file.js";
import type { TrajectoryEvent } from "../trajectory/types.js";
import { resolveSessionStoreTargetsOrExit } from "./session-store-targets.js";

type SessionDiagnoseOptions = {
  sessionKey?: string;
  store?: string;
  agent?: string;
  allAgents?: boolean;
  json?: boolean;
  brief?: boolean;
  limit?: string;
};

type StuckClassification =
  | "idle"
  | "model_call"
  | "tool_call"
  | "quota_suspended"
  | "delivery_pending"
  | "subagent_wedged"
  | "blocked"
  | "lock_held"
  | "processing"
  | "stale"
  | "unknown";

type DiagnosisEvent = {
  type: string;
  ts: string;
  preview?: string;
  event: TrajectoryEvent;
};

type SessionDiagnosis = {
  sessionKey: string;
  sessionId: string;
  agentId: string;
  classification: StuckClassification;
  summary: string;
  session: {
    updatedAt: number | undefined;
    ageMs: number | undefined;
    status: string | undefined;
    modelProvider: string | undefined;
    model: string | undefined;
    goal: { status: string; objective?: string; pausedAt?: number; blockedAt?: number } | undefined;
    quotaSuspension:
      | {
          reason?: string;
          state?: string;
          suspendedAt?: number;
          failedProvider?: string;
          failedModel?: string;
        }
      | undefined;
    pendingDelivery: boolean;
    subagentRecovery:
      | { wedgedAt?: number; wedgedReason?: string; automaticAttempts?: number }
      | undefined;
  };
  acp: {
    state: string | undefined;
    lastActivityAt: number | undefined;
    lastError: string | undefined;
  };
  trajectory: {
    filePath: string | undefined;
    lastEvents: DiagnosisEvent[];
    hasPendingPrompt: boolean;
    hasPendingToolCall: boolean;
    pendingToolName: string | undefined;
    lastModelCompletedAt: string | undefined;
    lastToolResultAt: string | undefined;
  };
  lock: {
    exists: boolean;
    pid: number | undefined;
    pidAlive: boolean | undefined;
    ageMs: number | undefined;
  };
  activity: {
    activeWorkKind: string | undefined;
    activeToolName: string | undefined;
    activeToolAgeMs: number | undefined;
    lastProgressAgeMs: number | undefined;
    lastProgressReason: string | undefined;
  };
};

const DEFAULT_EVENT_LIMIT = 50;
const RECENT_THRESHOLD_MS = 5 * 60_000; // 5 minutes
const STALE_THRESHOLD_MS = 30 * 60_000; // 30 minutes

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function toOptionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function isTrajectoryEvent(value: unknown): value is TrajectoryEvent {
  return (
    isRecord(value) &&
    value.traceSchema === "openclaw-trajectory" &&
    value.schemaVersion === 1 &&
    typeof value.type === "string" &&
    typeof value.ts === "string" &&
    typeof value.sessionId === "string"
  );
}

function parseTrajectoryEventLine(line: string): TrajectoryEvent | null {
  const trimmed = line.trim();
  if (!trimmed) {
    return null;
  }
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    return isTrajectoryEvent(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function readTrajectoryEvents(filePath: string, limit: number): TrajectoryEvent[] {
  try {
    const text = fs.readFileSync(filePath, "utf8");
    const lines = text.split(/\r?\n/u);
    const events: TrajectoryEvent[] = [];
    for (const line of lines) {
      const event = parseTrajectoryEventLine(line);
      if (event) {
        events.push(event);
      }
    }
    return events.slice(-limit);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return [];
    }
    throw error;
  }
}

function formatTimestamp(ts: string): string {
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) {
    return "--:--:--";
  }
  return date.toISOString().slice(11, 19);
}

function eventPreview(event: TrajectoryEvent): string {
  const data = event.data;
  switch (event.type) {
    case "session.started":
      return "session started";
    case "context.compiled": {
      const tools = Array.isArray(data?.tools) ? data.tools.length : undefined;
      return tools === undefined ? "context compiled" : `context compiled (${tools} tools)`;
    }
    case "prompt.submitted":
      return "prompt submitted";
    case "prompt.skipped": {
      const reason = toOptionalString(data?.reason);
      return `prompt skipped${reason ? `: ${reason}` : ""}`;
    }
    case "tool.call":
      return toOptionalString(data?.name) ?? toOptionalString(data?.toolName) ?? "tool call";
    case "tool.timeout":
      return `${toOptionalString(data?.name) ?? toOptionalString(data?.toolName) ?? "tool"} timeout`;
    case "tool.result": {
      const name = toOptionalString(data?.name) ?? toOptionalString(data?.toolName) ?? "tool";
      const status = data?.success === true ? "ok" : data?.success === false ? "error" : "done";
      return `${name} ${status}`;
    }
    case "model.completed": {
      const provider = event.provider?.trim();
      const model = event.modelId?.trim();
      const label = provider && model ? `${provider}/${model}` : model || provider || "model";
      const status =
        data?.timedOut === true ? "timeout" : data?.aborted === true ? "aborted" : "done";
      return `${label} ${status}`;
    }
    case "session.ended":
      return toOptionalString(data?.status) ?? "ended";
    default:
      return toOptionalString(data?.status) ?? toOptionalString(data?.name) ?? event.type;
  }
}

function toDiagnosisEvent(event: TrajectoryEvent): DiagnosisEvent {
  return {
    type: event.type,
    ts: event.ts,
    preview: eventPreview(event),
    event,
  };
}

type TrajectoryAnalysis = {
  lastEvents: DiagnosisEvent[];
  hasPendingPrompt: boolean;
  hasPendingToolCall: string | undefined;
  lastModelCompletedAt: string | undefined;
  lastToolResultAt: string | undefined;
};

function analyzeTrajectory(events: TrajectoryEvent[]): TrajectoryAnalysis {
  const diagnosisEvents = events.map(toDiagnosisEvent);

  let hasPendingPrompt = false;
  let hasPendingToolCall: string | undefined;
  let lastModelCompletedAt: string | undefined;
  let lastToolResultAt: string | undefined;

  // Walk events in chronological order to find unmatched prompt/tool calls
  let lastPromptSubmittedAt: string | undefined;
  let lastToolCallName: string | undefined;

  for (const event of events) {
    switch (event.type) {
      case "prompt.submitted":
        lastPromptSubmittedAt = event.ts;
        lastToolCallName = undefined;
        break;
      case "model.completed":
      case "model.call.error":
        lastPromptSubmittedAt = undefined;
        lastModelCompletedAt = event.ts;
        break;
      case "tool.call":
        lastToolCallName =
          toOptionalString(event.data?.name) ?? toOptionalString(event.data?.toolName);
        break;
      case "tool.result":
      case "tool.timeout":
        lastToolCallName = undefined;
        lastToolResultAt = event.ts;
        break;
      case "session.ended":
        // Session ended clears all pending state
        lastPromptSubmittedAt = undefined;
        lastToolCallName = undefined;
        break;
      default:
        break;
    }
  }

  hasPendingPrompt = lastPromptSubmittedAt !== undefined;
  if (lastToolCallName !== undefined) {
    hasPendingToolCall = lastToolCallName;
  }

  return {
    lastEvents: diagnosisEvents,
    hasPendingPrompt,
    hasPendingToolCall,
    lastModelCompletedAt,
    lastToolResultAt,
  };
}

type LockInfo = {
  exists: boolean;
  pid: number | undefined;
  pidAlive: boolean | undefined;
  ageMs: number | undefined;
};

function readLockInfo(sessionFile: string): LockInfo {
  const lockPath = `${sessionFile}.lock`;
  try {
    const text = fs.readFileSync(lockPath, "utf8");
    const parsed = JSON.parse(text) as unknown;
    if (!isRecord(parsed)) {
      return { exists: true, pid: undefined, pidAlive: undefined, ageMs: undefined };
    }
    const pid = typeof parsed.pid === "number" ? parsed.pid : undefined;
    const createdAt = typeof parsed.createdAt === "number" ? parsed.createdAt : undefined;
    const ageMs = createdAt ? Math.max(0, Date.now() - createdAt) : undefined;
    const alive = pid !== undefined ? isPidAlive(pid) : undefined;
    return { exists: true, pid, pidAlive: alive, ageMs };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return { exists: false, pid: undefined, pidAlive: undefined, ageMs: undefined };
    }
    return { exists: true, pid: undefined, pidAlive: undefined, ageMs: undefined };
  }
}

type DiagnosisCandidate = {
  agentId: string;
  key: string;
  entry: SessionEntry;
  storePath: string;
};

function selectCandidate(
  candidates: DiagnosisCandidate[],
  sessionKey?: string,
): DiagnosisCandidate | undefined {
  if (sessionKey) {
    const trimmed = sessionKey.trim();
    return candidates.find((c) => c.key === trimmed);
  }
  // Pick most recently active
  return candidates.toSorted((a, b) => (b.entry.updatedAt ?? 0) - (a.entry.updatedAt ?? 0))[0];
}

function classifySession(params: {
  entry: SessionEntry;
  acpState: string | undefined;
  acpLastActivityAt: number | undefined;
  trajectory: TrajectoryAnalysis;
  lock: LockInfo;
  activeWorkKind: string | undefined;
  lastProgressAgeMs: number | undefined;
}): { classification: StuckClassification; summary: string } {
  const {
    entry,
    acpState,
    acpLastActivityAt,
    trajectory,
    lock,
    activeWorkKind,
    lastProgressAgeMs,
  } = params;
  const now = Date.now();
  const ageMs = entry.updatedAt ? now - entry.updatedAt : undefined;
  const isRecent = ageMs !== undefined && ageMs < RECENT_THRESHOLD_MS;
  const isStale = ageMs !== undefined && ageMs > STALE_THRESHOLD_MS;
  const isTerminal = isTerminalSessionStatus(entry.status);

  // 1. Quota suspension
  if (entry.quotaSuspension && entry.quotaSuspension.state !== "resuming") {
    const reason = entry.quotaSuspension.reason ?? "rate limit";
    const provider = entry.quotaSuspension.failedProvider ?? "";
    const model = entry.quotaSuspension.failedModel ?? "";
    const target = provider || model ? ` (${provider}${provider && model ? "/" : ""}${model})` : "";
    return {
      classification: "quota_suspended",
      summary: `Session is suspended due to ${reason}${target}`,
    };
  }

  // 2. Subagent wedged
  if (entry.subagentRecovery?.wedgedAt) {
    const reason = entry.subagentRecovery.wedgedReason ?? "unknown reason";
    return {
      classification: "subagent_wedged",
      summary: `Subagent is wedged: ${reason}`,
    };
  }

  // 3. Goal blocked
  if (
    entry.goal?.status === "blocked" ||
    entry.goal?.status === "usage_limited" ||
    entry.goal?.status === "budget_limited"
  ) {
    return {
      classification: "blocked",
      summary: `Session goal is ${entry.goal.status}`,
    };
  }

  // 4. Pending delivery
  if (entry.pendingFinalDelivery) {
    return {
      classification: "delivery_pending",
      summary: "Pending final delivery after restart",
    };
  }

  // 5. Lock held by another process
  if (lock.exists && lock.pidAlive && lock.pid !== process.pid) {
    return {
      classification: "lock_held",
      summary: `Session lock held by PID ${lock.pid ?? "unknown"}`,
    };
  }

  // 6-8. Live diagnostic activity
  if (activeWorkKind === "model_call") {
    return {
      classification: "model_call",
      summary: "Waiting for model response",
    };
  }
  if (activeWorkKind === "tool_call") {
    return {
      classification: "tool_call",
      summary: "Executing a tool",
    };
  }
  if (
    activeWorkKind === "embedded_run" &&
    lastProgressAgeMs !== undefined &&
    lastProgressAgeMs < RECENT_THRESHOLD_MS
  ) {
    return {
      classification: "processing",
      summary: "Actively processing (embedded run in progress)",
    };
  }

  // 9-10. Trajectory-based detection
  if (trajectory.hasPendingPrompt) {
    return {
      classification: "model_call",
      summary: "Waiting for model response (prompt submitted, no completion recorded)",
    };
  }
  if (trajectory.hasPendingToolCall) {
    return {
      classification: "tool_call",
      summary: `Executing tool: ${trajectory.hasPendingToolCall}`,
    };
  }

  // 11-12. State-based detection
  if (acpState === "running" && acpLastActivityAt !== undefined) {
    const acpAge = now - acpLastActivityAt;
    if (acpAge < RECENT_THRESHOLD_MS) {
      return {
        classification: "processing",
        summary: "ACP runtime is actively processing",
      };
    }
  }
  if (entry.status === "running" && isRecent) {
    return {
      classification: "processing",
      summary: "Session is actively processing",
    };
  }

  // 13. Stale
  if (isStale && !isTerminal) {
    return {
      classification: "stale",
      summary: `No activity for ${formatAge(ageMs)}; session may be stuck`,
    };
  }

  // 14. Idle
  if (isTerminal || (entry.updatedAt !== undefined && !isRecent)) {
    return {
      classification: "idle",
      summary: "Session is idle",
    };
  }

  // 15. Fallback
  return {
    classification: "unknown",
    summary: "Unable to determine session state",
  };
}

function formatAge(ms: number | undefined): string {
  if (ms === undefined || ms < 0) {
    return "unknown";
  }
  if (ms < 1_000) {
    return `${ms}ms`;
  }
  if (ms < 60_000) {
    return `${Math.round(ms / 1_000)}s`;
  }
  if (ms < 3_600_000) {
    return `${Math.round(ms / 60_000)}m`;
  }
  if (ms < 86_400_000) {
    return `${Math.round(ms / 3_600_000)}h`;
  }
  return `${Math.round(ms / 86_400_000)}d`;
}

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) {
    return text;
  }
  return text.slice(0, maxLen - 3) + "...";
}

function formatTokens(usage: Record<string, unknown> | undefined): string {
  if (!usage || !isRecord(usage)) {
    return "";
  }
  const parts: string[] = [];
  if (typeof usage.input === "number" && usage.input > 0) {
    parts.push(`in:${usage.input}`);
  }
  if (typeof usage.output === "number" && usage.output > 0) {
    parts.push(`out:${usage.output}`);
  }
  if (typeof usage.cacheRead === "number" && usage.cacheRead > 0) {
    parts.push(`cache:${usage.cacheRead}`);
  }
  if (typeof usage.reasoningTokens === "number" && usage.reasoningTokens > 0) {
    parts.push(`reason:${usage.reasoningTokens}`);
  }
  if (typeof usage.total === "number" && usage.total > 0 && parts.length === 0) {
    parts.push(`total:${usage.total}`);
  }
  return parts.join(" ");
}

function computeDurationMs(startTs: string, endTs: string): number | undefined {
  const start = new Date(startTs).getTime();
  const end = new Date(endTs).getTime();
  if (Number.isNaN(start) || Number.isNaN(end) || end < start) {
    return undefined;
  }
  return end - start;
}

type Round = {
  startedAt: string | undefined;
  events: DiagnosisEvent[];
};

function groupIntoRounds(events: DiagnosisEvent[]): Round[] {
  const rounds: Round[] = [];
  let current: Round = { startedAt: undefined, events: [] };
  for (const ev of events) {
    if (ev.type === "session.started") {
      if (current.events.length > 0) {
        rounds.push(current);
      }
      current = { startedAt: ev.ts, events: [ev] };
    } else {
      current.events.push(ev);
    }
  }
  if (current.events.length > 0) {
    rounds.push(current);
  }
  return rounds;
}

function renderEventLine(
  ev: DiagnosisEvent,
  prevEvent: DiagnosisEvent | undefined,
  rich: boolean,
): string[] {
  const ts = formatTimestamp(ev.ts);
  const data = ev.event.data;
  const lines: string[] = [];
  const indent = "    ";

  switch (ev.type) {
    case "session.started": {
      const trigger = toOptionalString(data?.trigger) ?? "unknown";
      const toolCount = typeof data?.toolCount === "number" ? data.toolCount : undefined;
      lines.push(
        `${indent}${ts}  ${colorize(rich, theme.heading, "Session started")} (trigger: ${trigger}${toolCount !== undefined ? `, ${toolCount} tools` : ""})`,
      );
      break;
    }
    case "context.compiled": {
      const tools = Array.isArray(data?.tools) ? data.tools.length : undefined;
      const prompt = toOptionalString(data?.prompt);
      lines.push(
        `${indent}${ts}  Context compiled${tools !== undefined ? ` (${tools} tools)` : ""}`,
      );
      if (prompt) {
        lines.push(`${indent}      prompt: ${truncate(prompt.replace(/\s+/gu, " "), 120)}`);
      }
      break;
    }
    case "prompt.submitted": {
      const prompt = toOptionalString(data?.prompt);
      const images = typeof data?.imagesCount === "number" ? data.imagesCount : 0;
      lines.push(`${indent}${ts}  ${colorize(rich, theme.success, "Prompt submitted")}`);
      if (prompt) {
        lines.push(`${indent}      text: ${truncate(prompt.replace(/\s+/gu, " "), 200)}`);
      }
      if (images > 0) {
        lines.push(`${indent}      images: ${images}`);
      }
      break;
    }
    case "prompt.skipped": {
      const reason = toOptionalString(data?.reason) ?? "unknown";
      lines.push(`${indent}${ts}  ${colorize(rich, theme.warn, `Prompt skipped: ${reason}`)}`);
      break;
    }
    case "model.completed": {
      const provider = ev.event.provider?.trim();
      const model = ev.event.modelId?.trim();
      const label = provider && model ? `${provider}/${model}` : model || provider || "model";
      const isTimeout = data?.timedOut === true;
      const isAborted = data?.aborted === true;
      const hasError = toOptionalString(data?.promptError) ?? toOptionalString(data?.terminalError);
      let status: string;
      if (hasError) {
        status = colorize(rich, theme.error, `ERROR: ${hasError}`);
      } else if (isTimeout) {
        status = colorize(rich, theme.warn, "TIMEOUT");
      } else if (isAborted) {
        status = colorize(rich, theme.warn, "ABORTED");
      } else {
        status = colorize(rich, theme.success, "done");
      }

      // Duration from previous prompt.submitted
      let duration = "";
      if (prevEvent?.type === "prompt.submitted") {
        const ms = computeDurationMs(prevEvent.ts, ev.ts);
        if (ms !== undefined) {
          duration = ` (${formatAge(ms)})`;
        }
      }

      const tokens = formatTokens(data?.usage as Record<string, unknown> | undefined);
      lines.push(
        `${indent}${ts}  ${colorize(rich, theme.heading, "Model completed")} ${label} ${status}${duration}`,
      );
      if (tokens) {
        lines.push(`${indent}      tokens: ${tokens}`);
      }
      if (typeof data?.compactionCount === "number" && data.compactionCount > 0) {
        lines.push(`${indent}      compactions: ${data.compactionCount}`);
      }
      // Show assistant text preview
      const assistantTexts = Array.isArray(data?.assistantTexts) ? data.assistantTexts : [];
      if (assistantTexts.length > 0) {
        const text = assistantTexts.map((t: unknown) => (typeof t === "string" ? t : "")).join(" ");
        if (text.trim()) {
          lines.push(`${indent}      response: ${truncate(text.replace(/\s+/gu, " "), 200)}`);
        }
      }
      // Show error details
      if (data?.promptErrorSource) {
        lines.push(`${indent}      error source: ${data.promptErrorSource}`);
      }
      break;
    }
    case "model.call.error": {
      const error =
        toOptionalString(data?.error) ?? toOptionalString(data?.message) ?? "unknown error";
      lines.push(`${indent}${ts}  ${colorize(rich, theme.error, `Model call error: ${error}`)}`);
      break;
    }
    case "model.fallback_step": {
      const step = toOptionalString(data?.step) ?? toOptionalString(data?.status) ?? "fallback";
      lines.push(`${indent}${ts}  ${colorize(rich, theme.warn, `Model fallback: ${step}`)}`);
      break;
    }
    case "tool.call": {
      const name = toOptionalString(data?.name) ?? toOptionalString(data?.toolName) ?? "tool";
      const args = data?.arguments;
      let argsPreview = "";
      if (isRecord(args)) {
        argsPreview = truncate(JSON.stringify(args), 150);
      } else if (typeof args === "string") {
        argsPreview = truncate(args, 150);
      }
      lines.push(`${indent}${ts}  ${colorize(rich, theme.heading, `Tool call: ${name}`)}`);
      if (argsPreview) {
        lines.push(`${indent}      args: ${argsPreview}`);
      }
      break;
    }
    case "tool.result": {
      const name = toOptionalString(data?.name) ?? toOptionalString(data?.toolName) ?? "tool";
      const success = data?.success;
      const isError = success === false;
      const statusStr = isError
        ? colorize(rich, theme.error, "ERROR")
        : colorize(rich, theme.success, "ok");
      lines.push(`${indent}${ts}  Tool result: ${name} ${statusStr}`);
      // Show error message from result
      if (isError) {
        const errMsg = toOptionalString(data?.error) ?? toOptionalString(data?.message);
        if (errMsg) {
          lines.push(`${indent}      error: ${truncate(errMsg, 200)}`);
        }
      }
      break;
    }
    case "tool.timeout": {
      const name = toOptionalString(data?.name) ?? toOptionalString(data?.toolName) ?? "tool";
      lines.push(`${indent}${ts}  ${colorize(rich, theme.error, `Tool timeout: ${name}`)}`);
      break;
    }
    case "trace.artifacts": {
      const finalStatus = toOptionalString(data?.finalStatus);
      const toolMetas = Array.isArray(data?.toolMetas) ? data.toolMetas : [];
      const lastToolError = toOptionalString(data?.lastToolError);
      if (finalStatus && finalStatus !== "success") {
        lines.push(`${indent}${ts}  ${colorize(rich, theme.warn, `Artifacts: ${finalStatus}`)}`);
      }
      if (toolMetas.length > 0) {
        const toolNames = toolMetas
          .map((m: unknown) => (isRecord(m) ? (toOptionalString(m.toolName) ?? "?") : "?"))
          .join(", ");
        lines.push(`${indent}      tools used: ${toolNames}`);
      }
      if (lastToolError) {
        lines.push(
          `${indent}      ${colorize(rich, theme.error, `last tool error: ${truncate(lastToolError, 200)}`)}`,
        );
      }
      const artifactsTokens = formatTokens(data?.usage as Record<string, unknown> | undefined);
      if (artifactsTokens) {
        lines.push(`${indent}      total tokens: ${artifactsTokens}`);
      }
      break;
    }
    case "session.ended": {
      const status = toOptionalString(data?.status) ?? "ended";
      const isOk = status === "success";
      const statusColor = isOk ? theme.success : theme.error;
      const timedOut = data?.timedOut === true;
      const terminalError = toOptionalString(data?.terminalError);
      let detail = "";
      if (terminalError) {
        detail = ` - ${terminalError}`;
      } else if (timedOut) {
        detail = " (timed out)";
      }
      lines.push(
        `${indent}${ts}  ${colorize(rich, statusColor, `Session ended: ${status}${detail}`)}`,
      );
      break;
    }
    case "trace.metadata": {
      const harness = data?.harness;
      const modelInfo = data?.model;
      if (isRecord(modelInfo)) {
        const prov = toOptionalString(modelInfo.provider) ?? "";
        const name = toOptionalString(modelInfo.name) ?? "";
        const api = toOptionalString(modelInfo.api) ?? "";
        lines.push(`${indent}${ts}  Model: ${prov}/${name} (api: ${api})`);
      }
      if (isRecord(harness)) {
        const ver = toOptionalString(harness.version) ?? "";
        const sha = toOptionalString(harness.gitSha)?.slice(0, 8) ?? "";
        lines.push(`${indent}      harness: ${ver} (${sha})`);
      }
      break;
    }
    case "trace.truncated": {
      lines.push(
        `${indent}${ts}  ${colorize(rich, theme.warn, "Trajectory truncated (file size limit)")}`,
      );
      break;
    }
    default: {
      const preview = ev.preview ?? ev.type;
      lines.push(`${indent}${ts}  ${ev.type}: ${preview}`);
      break;
    }
  }
  return lines;
}

function hasRoundErrors(round: Round): boolean {
  for (const ev of round.events) {
    const data = ev.event.data;
    if (ev.type === "model.completed") {
      if (data?.timedOut || data?.aborted || data?.promptError || data?.terminalError) {
        return true;
      }
    }
    if (ev.type === "model.call.error" || ev.type === "tool.timeout") {
      return true;
    }
    if (ev.type === "tool.result" && data?.success === false) {
      return true;
    }
    if (ev.type === "session.ended" && data?.status !== "success") {
      return true;
    }
  }
  return false;
}

function renderBriefRound(round: Round, roundIdx: number): string[] {
  const lines: string[] = [];
  const ts = round.startedAt ? formatTimestamp(round.startedAt) : "??";
  lines.push(`  Round ${roundIdx + 1} (${ts}):`);
  for (const ev of round.events) {
    const data = ev.event.data;
    const evTs = formatTimestamp(ev.ts);
    if (ev.type === "model.completed") {
      const err = toOptionalString(data?.promptError) ?? toOptionalString(data?.terminalError);
      if (err) {
        lines.push(`    ${evTs}  model error: ${err}`);
      } else if (data?.timedOut) {
        lines.push(`    ${evTs}  model timeout`);
      } else if (data?.aborted) {
        lines.push(`    ${evTs}  model aborted`);
      }
    } else if (ev.type === "model.call.error") {
      lines.push(
        `    ${evTs}  model call error: ${toOptionalString(data?.error) ?? toOptionalString(data?.message) ?? "unknown"}`,
      );
    } else if (ev.type === "tool.result" && data?.success === false) {
      const name = toOptionalString(data?.name) ?? toOptionalString(data?.toolName) ?? "tool";
      const errMsg = toOptionalString(data?.error) ?? toOptionalString(data?.message) ?? "";
      lines.push(`    ${evTs}  ${name} error: ${truncate(errMsg, 200)}`);
    } else if (ev.type === "tool.timeout") {
      const name = toOptionalString(data?.name) ?? toOptionalString(data?.toolName) ?? "tool";
      lines.push(`    ${evTs}  ${name} timeout`);
    } else if (ev.type === "session.ended" && data?.status !== "success") {
      const err = toOptionalString(data?.terminalError) ?? "";
      lines.push(`    ${evTs}  ended: ${data?.status ?? "unknown"}${err ? ` - ${err}` : ""}`);
    }
  }
  return lines;
}

function classificationLabel(c: StuckClassification): string {
  switch (c) {
    case "idle":
      return "idle";
    case "model_call":
      return "model_call";
    case "tool_call":
      return "tool_call";
    case "quota_suspended":
      return "quota_suspended";
    case "delivery_pending":
      return "delivery_pending";
    case "subagent_wedged":
      return "subagent_wedged";
    case "blocked":
      return "blocked";
    case "lock_held":
      return "lock_held";
    case "processing":
      return "processing";
    case "stale":
      return "stale";
    case "unknown":
      return "unknown";
  }
}

function isStuckClassification(c: StuckClassification): boolean {
  return (
    c === "model_call" ||
    c === "tool_call" ||
    c === "quota_suspended" ||
    c === "delivery_pending" ||
    c === "subagent_wedged" ||
    c === "blocked" ||
    c === "lock_held" ||
    c === "stale"
  );
}

function colorizeClassification(label: string, c: StuckClassification, rich: boolean): string {
  if (!rich) {
    return label;
  }
  if (c === "processing" || c === "idle") {
    return theme.success(label);
  }
  if (isStuckClassification(c)) {
    return theme.warn(label);
  }
  return theme.muted(label);
}

function writeTextOutput(diagnosis: SessionDiagnosis, runtime: RuntimeEnv): void {
  const rich = isRich();
  const classLabel = classificationLabel(diagnosis.classification);
  const coloredClass = colorizeClassification(classLabel, diagnosis.classification, rich);

  runtime.log(colorize(rich, theme.heading, `Session Diagnosis: ${diagnosis.sessionKey}`));
  runtime.log(colorize(rich, theme.heading, "=".repeat(50)));
  runtime.log(`Classification: ${coloredClass} -- ${diagnosis.summary}`);
  runtime.log("");

  // Session section
  runtime.log(colorize(rich, theme.heading, "Session"));
  const s = diagnosis.session;
  runtime.log(`  Status:      ${s.status ?? "unknown"}`);
  runtime.log(`  Updated:     ${s.ageMs !== undefined ? `${formatAge(s.ageMs)} ago` : "unknown"}`);
  if (s.modelProvider || s.model) {
    runtime.log(
      `  Model:       ${s.modelProvider ?? ""}${s.modelProvider && s.model ? "/" : ""}${s.model ?? ""}`,
    );
  }
  if (s.goal) {
    runtime.log(
      `  Goal:        ${s.goal.status}${s.goal.objective ? ` - "${s.goal.objective}"` : ""}`,
    );
  }
  if (s.quotaSuspension) {
    runtime.log(
      `  Suspension:  ${s.quotaSuspension.state ?? "suspended"} (${s.quotaSuspension.reason ?? "rate limit"})`,
    );
  }
  if (s.pendingDelivery) {
    runtime.log(`  Delivery:    pending final delivery`);
  }
  if (s.subagentRecovery) {
    runtime.log(`  Recovery:    wedged (${s.subagentRecovery.wedgedReason ?? "unknown"})`);
  }
  runtime.log("");

  // ACP section
  const acp = diagnosis.acp;
  if (acp.state || acp.lastActivityAt || acp.lastError) {
    runtime.log(colorize(rich, theme.heading, "ACP Runtime"));
    runtime.log(`  State:       ${acp.state ?? "n/a"}`);
    if (acp.lastActivityAt) {
      runtime.log(`  Last active: ${formatAge(Date.now() - acp.lastActivityAt)} ago`);
    }
    if (acp.lastError) {
      runtime.log(`  Last error:  ${acp.lastError}`);
    }
    runtime.log("");
  }

  // Trajectory section - grouped by rounds
  const t = diagnosis.trajectory;
  const rounds = groupIntoRounds(t.lastEvents);
  runtime.log(
    colorize(
      rich,
      theme.heading,
      `Execution Timeline (${t.lastEvents.length} events, ${rounds.length} rounds)`,
    ),
  );
  if (rounds.length === 0) {
    runtime.log("  No trajectory events found");
  } else {
    for (let ri = 0; ri < rounds.length; ri++) {
      const round = rounds[ri];
      const roundTs = round.startedAt ? formatTimestamp(round.startedAt) : "??";
      runtime.log("");
      runtime.log(colorize(rich, theme.heading, `  --- Round ${ri + 1} (${roundTs}) ---`));
      for (let ei = 0; ei < round.events.length; ei++) {
        const ev = round.events[ei];
        const prev = ei > 0 ? round.events[ei - 1] : undefined;
        const rendered = renderEventLine(ev, prev, rich);
        for (const line of rendered) {
          runtime.log(line);
        }
      }
      // Show pending markers at round level
      if (t.hasPendingPrompt && ri === rounds.length - 1) {
        runtime.log(colorize(rich, theme.warn, "    <-- waiting for model response"));
      }
      if (t.hasPendingToolCall && ri === rounds.length - 1) {
        runtime.log(
          colorize(rich, theme.warn, `    <-- waiting for tool: ${t.pendingToolName ?? "unknown"}`),
        );
      }
    }
  }
  runtime.log("");

  // Lock section
  runtime.log(colorize(rich, theme.heading, "Lock File"));
  if (diagnosis.lock.exists) {
    runtime.log(`  PID:         ${diagnosis.lock.pid ?? "unknown"}`);
    runtime.log(
      `  PID alive:   ${diagnosis.lock.pidAlive === true ? "yes" : diagnosis.lock.pidAlive === false ? "no" : "unknown"}`,
    );
    if (diagnosis.lock.ageMs !== undefined) {
      runtime.log(`  Age:         ${formatAge(diagnosis.lock.ageMs)}`);
    }
  } else {
    runtime.log("  Not present");
  }
  runtime.log("");

  // Diagnostic activity section
  const a = diagnosis.activity;
  if (a.activeWorkKind || a.lastProgressAgeMs !== undefined) {
    runtime.log(colorize(rich, theme.heading, "Diagnostic Activity"));
    if (a.activeWorkKind) {
      runtime.log(
        `  Active work:  ${a.activeWorkKind}${a.activeToolName ? ` (${a.activeToolName})` : ""}`,
      );
    }
    if (a.activeToolAgeMs !== undefined) {
      runtime.log(`  Tool age:     ${formatAge(a.activeToolAgeMs)}`);
    }
    if (a.lastProgressAgeMs !== undefined) {
      runtime.log(
        `  Last progress: ${formatAge(a.lastProgressAgeMs)} ago${a.lastProgressReason ? ` (${a.lastProgressReason})` : ""}`,
      );
    }
  } else {
    runtime.log(colorize(rich, theme.heading, "Diagnostic Activity"));
    runtime.log("  No active diagnostic data (session may not be in this process)");
  }
}

function writeBriefOutput(diagnosis: SessionDiagnosis, runtime: RuntimeEnv): void {
  const rich = isRich();
  const classLabel = classificationLabel(diagnosis.classification);
  const coloredClass = colorizeClassification(classLabel, diagnosis.classification, rich);

  runtime.log(colorize(rich, theme.heading, `Session Diagnosis: ${diagnosis.sessionKey}`));
  runtime.log(`Classification: ${coloredClass} -- ${diagnosis.summary}`);
  runtime.log("");

  const s = diagnosis.session;
  if (s.quotaSuspension) {
    runtime.log(
      colorize(
        rich,
        theme.error,
        `  Quota suspended: ${s.quotaSuspension.reason ?? "rate limit"} (${s.quotaSuspension.failedProvider ?? ""}${s.quotaSuspension.failedModel ? "/" : ""}${s.quotaSuspension.failedModel ?? ""})`,
      ),
    );
  }
  if (s.subagentRecovery) {
    runtime.log(
      colorize(
        rich,
        theme.error,
        `  Subagent wedged: ${s.subagentRecovery.wedgedReason ?? "unknown"}`,
      ),
    );
  }
  if (
    s.goal &&
    (s.goal.status === "blocked" ||
      s.goal.status === "usage_limited" ||
      s.goal.status === "budget_limited")
  ) {
    runtime.log(colorize(rich, theme.warn, `  Goal ${s.goal.status}`));
  }

  // Show only error rounds from trajectory
  const t = diagnosis.trajectory;
  const rounds = groupIntoRounds(t.lastEvents);
  const errorRounds = rounds.filter((r) => hasRoundErrors(r));
  if (errorRounds.length > 0) {
    runtime.log("");
    runtime.log(
      colorize(rich, theme.heading, `Error Rounds (${errorRounds.length} of ${rounds.length}):`),
    );
    for (let i = 0; i < rounds.length; i++) {
      if (hasRoundErrors(rounds[i])) {
        const lines = renderBriefRound(rounds[i], i);
        for (const line of lines) {
          runtime.log(line);
        }
      }
    }
  } else if (rounds.length > 0) {
    runtime.log("");
    runtime.log(
      colorize(rich, theme.success, `  All ${rounds.length} rounds completed without errors.`),
    );
  }

  // Lock
  if (diagnosis.lock.exists && diagnosis.lock.pidAlive) {
    runtime.log("");
    runtime.log(
      colorize(
        rich,
        theme.warn,
        `  Lock held by PID ${diagnosis.lock.pid ?? "unknown"} (${formatAge(diagnosis.lock.ageMs)} ago)`,
      ),
    );
  }
}

export async function sessionsDiagnoseCommand(
  opts: SessionDiagnoseOptions,
  runtime: RuntimeEnv,
): Promise<void> {
  const limitRaw = opts.limit?.trim();
  let eventLimit = DEFAULT_EVENT_LIMIT;
  if (limitRaw !== undefined) {
    const parsed = Number.parseInt(limitRaw, 10);
    if (!Number.isFinite(parsed) || parsed < 0) {
      runtime.error("--limit must be a non-negative integer.");
      runtime.exit(1);
      return;
    }
    eventLimit = parsed;
  }

  const cfg = getRuntimeConfig();
  const targets = resolveSessionStoreTargetsOrExit({
    cfg,
    opts: {
      store: opts.store,
      agent: opts.agent,
      allAgents: opts.allAgents,
    },
    runtime,
  });
  if (!targets) {
    return;
  }

  // Gather all candidates
  const candidates: DiagnosisCandidate[] = [];
  for (const target of targets) {
    const store = loadSessionStore(target.storePath);
    for (const [key, entry] of Object.entries(store)) {
      candidates.push({
        agentId: target.agentId,
        key,
        entry,
        storePath: target.storePath,
      });
    }
  }

  const candidate = selectCandidate(candidates, opts.sessionKey);
  if (!candidate) {
    const suffix = opts.sessionKey ? ` for key: ${opts.sessionKey}` : "";
    runtime.error(`No session found${suffix}.`);
    runtime.exit(1);
    return;
  }

  const { agentId, key, entry, storePath } = candidate;
  const sessionsDir = path.dirname(storePath);

  // Resolve trajectory file path
  const sessionFile = resolveSessionFilePath(entry.sessionId, entry, {
    agentId,
    sessionsDir,
  });
  let trajectoryFilePath: string | undefined;
  try {
    trajectoryFilePath =
      (await resolveTrajectoryRuntimeFile({
        sessionFile,
        sessionId: entry.sessionId,
      })) ??
      resolveTrajectoryFilePath({
        sessionFile,
        sessionId: entry.sessionId,
      });
  } catch {
    // trajectory file resolution failed; continue without it
  }

  // Gather data from each source
  const acpSessionKey = resolveStoredSessionKeyForAgentStore({
    cfg,
    agentId,
    sessionKey: key,
  });

  let acpMeta: { state?: string; lastActivityAt?: number; lastError?: string } | undefined;
  try {
    const meta = readAcpSessionMetaForEntry({ sessionKey: acpSessionKey, entry });
    if (meta) {
      acpMeta = {
        state: meta.state,
        lastActivityAt: meta.lastActivityAt,
        lastError: meta.lastError,
      };
    }
  } catch {
    // ACP metadata unavailable
  }

  // Read trajectory events
  let trajectoryAnalysis: TrajectoryAnalysis;
  try {
    const events = trajectoryFilePath ? readTrajectoryEvents(trajectoryFilePath, eventLimit) : [];
    trajectoryAnalysis = analyzeTrajectory(events);
  } catch {
    trajectoryAnalysis = {
      lastEvents: [],
      hasPendingPrompt: false,
      hasPendingToolCall: undefined,
      lastModelCompletedAt: undefined,
      lastToolResultAt: undefined,
    };
  }

  // Read lock info
  let lockInfo: LockInfo;
  try {
    lockInfo = readLockInfo(sessionFile);
  } catch {
    lockInfo = { exists: false, pid: undefined, pidAlive: undefined, ageMs: undefined };
  }

  // Read diagnostic activity
  let activeWorkKind: string | undefined;
  let activeToolName: string | undefined;
  let activeToolAgeMs: number | undefined;
  let lastProgressAgeMs: number | undefined;
  let lastProgressReason: string | undefined;
  try {
    const snapshot = getDiagnosticSessionActivitySnapshot({
      sessionId: entry.sessionId,
      sessionKey: key,
    });
    activeWorkKind = snapshot.activeWorkKind;
    activeToolName = snapshot.activeToolName;
    activeToolAgeMs = snapshot.activeToolAgeMs;
    lastProgressAgeMs = snapshot.lastProgressAgeMs;
    lastProgressReason = snapshot.lastProgressReason;
  } catch {
    // Diagnostic activity unavailable
  }

  // Classify
  const { classification, summary } = classifySession({
    entry,
    acpState: acpMeta?.state,
    acpLastActivityAt: acpMeta?.lastActivityAt,
    trajectory: trajectoryAnalysis,
    lock: lockInfo,
    activeWorkKind,
    lastProgressAgeMs,
  });

  const now = Date.now();
  const diagnosis: SessionDiagnosis = {
    sessionKey: key,
    sessionId: entry.sessionId,
    agentId,
    classification,
    summary,
    session: {
      updatedAt: entry.updatedAt,
      ageMs: entry.updatedAt ? now - entry.updatedAt : undefined,
      status: entry.status,
      modelProvider: entry.modelProvider,
      model: entry.model,
      goal: entry.goal
        ? {
            status: entry.goal.status,
            objective: entry.goal.objective,
            pausedAt: entry.goal.pausedAt,
            blockedAt: entry.goal.blockedAt,
          }
        : undefined,
      quotaSuspension: entry.quotaSuspension
        ? {
            reason: entry.quotaSuspension.reason,
            state: entry.quotaSuspension.state,
            suspendedAt: entry.quotaSuspension.suspendedAt,
            failedProvider: entry.quotaSuspension.failedProvider,
            failedModel: entry.quotaSuspension.failedModel,
          }
        : undefined,
      pendingDelivery: entry.pendingFinalDelivery === true,
      subagentRecovery: entry.subagentRecovery
        ? {
            wedgedAt: entry.subagentRecovery.wedgedAt,
            wedgedReason: entry.subagentRecovery.wedgedReason,
            automaticAttempts: entry.subagentRecovery.automaticAttempts,
          }
        : undefined,
    },
    acp: {
      state: acpMeta?.state,
      lastActivityAt: acpMeta?.lastActivityAt,
      lastError: acpMeta?.lastError,
    },
    trajectory: {
      filePath: trajectoryFilePath,
      lastEvents: trajectoryAnalysis.lastEvents,
      hasPendingPrompt: trajectoryAnalysis.hasPendingPrompt,
      hasPendingToolCall: trajectoryAnalysis.hasPendingToolCall !== undefined,
      pendingToolName: trajectoryAnalysis.hasPendingToolCall,
      lastModelCompletedAt: trajectoryAnalysis.lastModelCompletedAt,
      lastToolResultAt: trajectoryAnalysis.lastToolResultAt,
    },
    lock: lockInfo,
    activity: {
      activeWorkKind,
      activeToolName,
      activeToolAgeMs,
      lastProgressAgeMs,
      lastProgressReason,
    },
  };

  if (opts.json) {
    writeRuntimeJson(runtime, diagnosis);
    return;
  }

  if (opts.brief) {
    writeBriefOutput(diagnosis, runtime);
  } else {
    writeTextOutput(diagnosis, runtime);
  }
}

export const testing = {
  classifySession,
  analyzeTrajectory,
  formatAge,
  readLockInfo,
  selectCandidate,
} as const;
export { testing as __testing };
