import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RuntimeEnv } from "../runtime.js";
import type { TrajectoryEvent } from "../trajectory/types.js";
import { sessionsDiagnoseCommand } from "./sessions-diagnose.js";

const mocks = vi.hoisted(() => ({
  getRuntimeConfig: vi.fn(() => ({})),
  readAcpSessionMetaForEntry: vi.fn(() => undefined),
  getDiagnosticSessionActivitySnapshot: vi.fn(() => ({})),
  isPidAlive: vi.fn((_pid?: number) => false),
}));

vi.mock("../config/config.js", () => ({
  getRuntimeConfig: mocks.getRuntimeConfig,
}));

vi.mock("../acp/runtime/session-meta.js", () => ({
  readAcpSessionMetaForEntry: mocks.readAcpSessionMetaForEntry,
}));

vi.mock("../logging/diagnostic-run-activity.js", () => ({
  getDiagnosticSessionActivitySnapshot: mocks.getDiagnosticSessionActivitySnapshot,
}));

vi.mock("../shared/pid-alive.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../shared/pid-alive.js")>();
  return {
    ...actual,
    isPidAlive: mocks.isPidAlive,
  };
});

const sessionKey = "agent:main:telegram:direct:owner";

function makeRuntime(): RuntimeEnv {
  return {
    log: vi.fn(),
    error: vi.fn(),
    exit: vi.fn(),
  };
}

function makeEvent(
  params: Partial<TrajectoryEvent> & { type: string; ts: string },
): TrajectoryEvent {
  return {
    traceSchema: "openclaw-trajectory",
    schemaVersion: 1,
    traceId: "trace-1",
    source: "runtime",
    seq: 1,
    sessionId: "session-one",
    sessionKey,
    ...params,
  };
}

function writeJsonl(filePath: string, events: TrajectoryEvent[]): void {
  fs.writeFileSync(filePath, `${events.map((event) => JSON.stringify(event)).join("\n")}\n`);
}

function runtimeOutput(runtime: RuntimeEnv): string {
  return vi
    .mocked(runtime.log)
    .mock.calls.map((call) => String(call[0]))
    .join("\n");
}

describe("sessionsDiagnoseCommand", () => {
  let tmpDir: string;
  let storePath: string;
  let previousStateDir: string | undefined;

  beforeEach(() => {
    previousStateDir = process.env.OPENCLAW_STATE_DIR;
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "openclaw-sessions-diagnose-"));
    process.env.OPENCLAW_STATE_DIR = path.join(tmpDir, "state");
    mocks.getRuntimeConfig.mockReturnValue({
      agents: {
        list: [{ id: "main" }],
      },
    });
    storePath = path.join(tmpDir, "sessions.json");
    mocks.readAcpSessionMetaForEntry.mockReturnValue(undefined);
    mocks.getDiagnosticSessionActivitySnapshot.mockReturnValue({});
    mocks.isPidAlive.mockReturnValue(false);
  });

  afterEach(() => {
    if (previousStateDir !== undefined) {
      process.env.OPENCLAW_STATE_DIR = previousStateDir;
    } else {
      delete process.env.OPENCLAW_STATE_DIR;
    }
    fs.rmSync(tmpDir, { recursive: true, force: true });
    vi.restoreAllMocks();
  });

  function writeStore(entries: Record<string, Record<string, unknown>>): void {
    fs.writeFileSync(storePath, `${JSON.stringify(entries)}\n`);
  }

  it("classifies idle session when last activity was long ago", async () => {
    writeStore({
      [sessionKey]: {
        sessionId: "session-one",
        sessionFile: "session-one.jsonl",
        updatedAt: Date.now() - 2 * 60 * 60_000, // 2 hours ago
        status: "done",
      },
    });

    const runtime = makeRuntime();
    await sessionsDiagnoseCommand({ store: storePath }, runtime);

    const output = runtimeOutput(runtime);
    expect(output).toContain("idle");
  });

  it("classifies model_call when trajectory shows prompt.submitted without model.completed", async () => {
    const now = new Date().toISOString();
    writeStore({
      [sessionKey]: {
        sessionId: "session-one",
        sessionFile: "session-one.jsonl",
        updatedAt: Date.now(),
        status: "running",
      },
    });
    const trajectoryPath = path.join(tmpDir, "session-one.trajectory.jsonl");
    writeJsonl(trajectoryPath, [
      makeEvent({ type: "session.started", ts: now, seq: 1 }),
      makeEvent({ type: "prompt.submitted", ts: now, seq: 2 }),
    ]);

    const runtime = makeRuntime();
    await sessionsDiagnoseCommand({ store: storePath }, runtime);

    const output = runtimeOutput(runtime);
    expect(output).toContain("model_call");
    expect(output).toContain("Waiting for model response");
  });

  it("classifies tool_call when trajectory shows tool.call without tool.result", async () => {
    const now = new Date().toISOString();
    writeStore({
      [sessionKey]: {
        sessionId: "session-one",
        sessionFile: "session-one.jsonl",
        updatedAt: Date.now(),
        status: "running",
      },
    });
    const trajectoryPath = path.join(tmpDir, "session-one.trajectory.jsonl");
    writeJsonl(trajectoryPath, [
      makeEvent({ type: "session.started", ts: now, seq: 1 }),
      makeEvent({ type: "prompt.submitted", ts: now, seq: 2 }),
      makeEvent({ type: "model.completed", ts: now, seq: 3 }),
      makeEvent({ type: "tool.call", ts: now, seq: 4, data: { name: "bash" } }),
    ]);

    const runtime = makeRuntime();
    await sessionsDiagnoseCommand({ store: storePath }, runtime);

    const output = runtimeOutput(runtime);
    expect(output).toContain("tool_call");
    expect(output).toContain("bash");
  });

  it("classifies quota_suspended when session has quotaSuspension", async () => {
    writeStore({
      [sessionKey]: {
        sessionId: "session-one",
        sessionFile: "session-one.jsonl",
        updatedAt: Date.now(),
        status: "running",
        quotaSuspension: {
          state: "suspended",
          reason: "rate_limit_exceeded",
          suspendedAt: Date.now() - 60_000,
          failedProvider: "openai",
          failedModel: "gpt-4o",
        },
      },
    });

    const runtime = makeRuntime();
    await sessionsDiagnoseCommand({ store: storePath }, runtime);

    const output = runtimeOutput(runtime);
    expect(output).toContain("quota_suspended");
    expect(output).toContain("rate_limit_exceeded");
  });

  it("classifies delivery_pending when session has pendingFinalDelivery", async () => {
    writeStore({
      [sessionKey]: {
        sessionId: "session-one",
        sessionFile: "session-one.jsonl",
        updatedAt: Date.now(),
        status: "running",
        pendingFinalDelivery: true,
      },
    });

    const runtime = makeRuntime();
    await sessionsDiagnoseCommand({ store: storePath }, runtime);

    const output = runtimeOutput(runtime);
    expect(output).toContain("delivery_pending");
    expect(output).toContain("Pending final delivery");
  });

  it("classifies subagent_wedged when subagentRecovery has wedgedAt", async () => {
    writeStore({
      [sessionKey]: {
        sessionId: "session-one",
        sessionFile: "session-one.jsonl",
        updatedAt: Date.now(),
        status: "running",
        subagentRecovery: {
          wedgedAt: Date.now() - 120_000,
          wedgedReason: "no_output_timeout",
          automaticAttempts: 2,
        },
      },
    });

    const runtime = makeRuntime();
    await sessionsDiagnoseCommand({ store: storePath }, runtime);

    const output = runtimeOutput(runtime);
    expect(output).toContain("subagent_wedged");
    expect(output).toContain("no_output_timeout");
  });

  it("classifies blocked when goal status is usage_limited", async () => {
    writeStore({
      [sessionKey]: {
        sessionId: "session-one",
        sessionFile: "session-one.jsonl",
        updatedAt: Date.now(),
        status: "running",
        goal: {
          status: "usage_limited",
          objective: "Help user",
        },
      },
    });

    const runtime = makeRuntime();
    await sessionsDiagnoseCommand({ store: storePath }, runtime);

    const output = runtimeOutput(runtime);
    expect(output).toContain("blocked");
    expect(output).toContain("usage_limited");
  });

  it("classifies processing when recently updated and running", async () => {
    writeStore({
      [sessionKey]: {
        sessionId: "session-one",
        sessionFile: "session-one.jsonl",
        updatedAt: Date.now() - 30_000, // 30 seconds ago
        status: "running",
      },
    });

    const runtime = makeRuntime();
    await sessionsDiagnoseCommand({ store: storePath }, runtime);

    const output = runtimeOutput(runtime);
    expect(output).toContain("processing");
  });

  it("outputs valid JSON with --json", async () => {
    writeStore({
      [sessionKey]: {
        sessionId: "session-one",
        sessionFile: "session-one.jsonl",
        updatedAt: Date.now(),
        status: "running",
      },
    });

    const runtime = makeRuntime();
    await sessionsDiagnoseCommand({ store: storePath, json: true }, runtime);

    const jsonCall = vi.mocked(runtime.log).mock.calls.find((call) => {
      try {
        const parsed = JSON.parse(String(call[0]));
        return parsed && typeof parsed.classification === "string";
      } catch {
        return false;
      }
    });
    expect(jsonCall).toBeDefined();
    const parsed = JSON.parse(String(jsonCall![0]));
    expect(parsed).toHaveProperty("sessionKey", sessionKey);
    expect(parsed).toHaveProperty("sessionId", "session-one");
    expect(parsed).toHaveProperty("classification");
    expect(parsed).toHaveProperty("summary");
    expect(parsed).toHaveProperty("session");
    expect(parsed).toHaveProperty("trajectory");
    expect(parsed).toHaveProperty("lock");
    expect(parsed).toHaveProperty("activity");
  });

  it("handles missing session gracefully", async () => {
    writeStore({});

    const runtime = makeRuntime();
    await sessionsDiagnoseCommand({ store: storePath, sessionKey: "nonexistent" }, runtime);

    expect(vi.mocked(runtime.error).mock.calls.length).toBeGreaterThan(0);
    expect(vi.mocked(runtime.exit).mock.calls).toContainEqual([1]);
  });

  it("handles missing trajectory file gracefully", async () => {
    writeStore({
      [sessionKey]: {
        sessionId: "session-one",
        sessionFile: "session-one.jsonl",
        updatedAt: Date.now() - 60_000,
        status: "running",
      },
    });
    // No trajectory file written

    const runtime = makeRuntime();
    await sessionsDiagnoseCommand({ store: storePath }, runtime);

    const output = runtimeOutput(runtime);
    expect(output).toContain("Session Diagnosis");
    expect(output).toContain("No trajectory events found");
  });

  it("selects most recently active session when no key specified", async () => {
    const key1 = "agent:main:telegram:direct:user1";
    const key2 = "agent:main:telegram:direct:user2";
    writeStore({
      [key1]: {
        sessionId: "session-old",
        sessionFile: "session-old.jsonl",
        updatedAt: Date.now() - 600_000, // 10 min ago
        status: "done",
      },
      [key2]: {
        sessionId: "session-new",
        sessionFile: "session-new.jsonl",
        updatedAt: Date.now() - 30_000, // 30s ago
        status: "running",
      },
    });

    const runtime = makeRuntime();
    await sessionsDiagnoseCommand({ store: storePath }, runtime);

    const output = runtimeOutput(runtime);
    expect(output).toContain(key2);
  });

  it("classifies lock_held when lock exists with alive PID", async () => {
    writeStore({
      [sessionKey]: {
        sessionId: "session-one",
        sessionFile: "session-one.jsonl",
        updatedAt: Date.now(),
        status: "running",
      },
    });

    // Write a lock file
    const lockPath = path.join(tmpDir, "session-one.jsonl.lock");
    fs.writeFileSync(lockPath, JSON.stringify({ pid: 99999, createdAt: Date.now() }));

    mocks.isPidAlive.mockReturnValue(true);

    const runtime = makeRuntime();
    await sessionsDiagnoseCommand({ store: storePath }, runtime);

    const output = runtimeOutput(runtime);
    expect(output).toContain("lock_held");
    expect(output).toContain("99999");
  });

  it("uses live diagnostic activity when available", async () => {
    writeStore({
      [sessionKey]: {
        sessionId: "session-one",
        sessionFile: "session-one.jsonl",
        updatedAt: Date.now(),
        status: "running",
      },
    });

    mocks.getDiagnosticSessionActivitySnapshot.mockReturnValue({
      activeWorkKind: "model_call",
      lastProgressAgeMs: 5000,
      lastProgressReason: "model_call:started",
    });

    const runtime = makeRuntime();
    await sessionsDiagnoseCommand({ store: storePath }, runtime);

    const output = runtimeOutput(runtime);
    expect(output).toContain("model_call");
  });
});
