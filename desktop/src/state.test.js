import { describe, expect, it } from "vitest";

import { createInitialState, reduceAction } from "./state";

describe("desktop state", () => {
  it("tracks setup steps, logs, and refresh state independently", () => {
    let state = createInitialState();
    state = reduceAction(state, {
      type: "bootstrap-snapshot",
      payload: {
        status: "running",
        onboarding_required: true,
        currentStep: 1,
        totalSteps: 6,
        message: "Preparing bundled runtime files",
        steps: state.setup.steps.map((step) =>
          step.task === "sync-runtime" ? { ...step, status: "running" } : step
        ),
        logs: ["Starting SemanticGallery desktop runtime."]
      }
    });
    state = reduceAction(state, { type: "setup-log", payload: { line: "Downloading weights" } });
    state = reduceAction(state, {
      type: "index-progress",
      payload: {
        status: "running",
        phase: "progress",
        current: 3,
        total: 10,
        startedAtMs: 1234,
        elapsedSeconds: 12,
        remainingSeconds: 28,
        message: "Indexing cat.jpg (3/10)"
      }
    });
    state = reduceAction(state, {
      type: "stage2-progress",
      payload: {
        status: "running",
        phase: "adapt",
        current: 4,
        total: 12,
        startedAtMs: 5678,
        elapsedSeconds: 30,
        remainingSeconds: 60,
        message: "Training epoch 1 of 1 · step 2 of 10"
      }
    });
    state = reduceAction(state, { type: "refresh-started" });

    expect(state.setup.currentStep).toBe(1);
    expect(state.setup.totalSteps).toBe(6);
    expect(state.setup.onboardingRequired).toBe(true);
    expect(state.setup.steps.find((step) => step.task === "sync-runtime")?.status).toBe("running");
    expect(state.setup.logs).toEqual([
      "Starting SemanticGallery desktop runtime.",
      "Downloading weights"
    ]);
    expect(state.indexing.current).toBe(3);
    expect(state.indexing.total).toBe(10);
    expect(state.indexing.startedAtMs).toBe(1234);
    expect(state.indexing.elapsedSeconds).toBe(12);
    expect(state.indexing.remainingSeconds).toBe(28);
    expect(state.indexing.message).toBe("Indexing cat.jpg (3/10)");
    expect(state.stage2.current).toBe(4);
    expect(state.stage2.total).toBe(12);
    expect(state.stage2.startedAtMs).toBe(5678);
    expect(state.stage2.elapsedSeconds).toBe(30);
    expect(state.stage2.remainingSeconds).toBe(60);
    expect(state.stage2.message).toBe("Training epoch 1 of 1 · step 2 of 10");
    expect(state.lastTaskMessage).toBe("Training epoch 1 of 1 · step 2 of 10");
    expect(state.refresh.isRunning).toBe(true);
  });
});
