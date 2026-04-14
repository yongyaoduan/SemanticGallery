import { describe, expect, it } from "vitest";

import { buildIndexProgressTiming, buildIndexProgressSummary, buildStage2ProgressSummary } from "./index-progress";

describe("index progress helpers", () => {
  it("formats elapsed and remaining time for an active indexing run", () => {
    expect(
      buildIndexProgressTiming({
        status: "running",
        startedAtMs: Date.parse("2026-04-13T15:00:00+08:00"),
        elapsedSeconds: 42,
        remainingSeconds: 93
      })
    ).toEqual({
      elapsedLabel: "Elapsed 00:42",
      remainingLabel: "Left 01:33",
      etaLabel: "Done by 15:02"
    });
  });

  it("keeps completed indexing runs at full width and clears the remaining time", () => {
    expect(
      buildIndexProgressSummary({
        status: "ready",
        current: 7062,
        total: 7062,
        startedAtMs: Date.parse("2026-04-13T15:00:00+08:00"),
        elapsedSeconds: 315,
        remainingSeconds: 0
      })
    ).toEqual({
      countLabel: "7062 / 7062",
      percent: 100,
      showProgress: false,
      indeterminate: false,
      elapsedLabel: "Elapsed 05:15",
      remainingLabel: "",
      etaLabel: ""
    });
  });

  it("shows the index progress shell only while indexing is running", () => {
    expect(
      buildIndexProgressSummary({
        status: "idle",
        current: 0,
        total: 0,
        message: "No indexing task is running."
      })
    ).toEqual({
      countLabel: "0 / 0",
      percent: 0,
      showProgress: false,
      indeterminate: false,
      elapsedLabel: "",
      remainingLabel: "",
      etaLabel: ""
    });

    expect(
      buildIndexProgressSummary({
        status: "running",
        current: 42,
        total: 7062,
        elapsedSeconds: 18,
        remainingSeconds: 120,
        message: "Indexing IMG_20230516_121154.jpg (42/7062)"
      })
    ).toEqual({
      countLabel: "42 / 7062",
      percent: 1,
      showProgress: true,
      indeterminate: false,
      elapsedLabel: "Elapsed 00:18",
      remainingLabel: "Left 02:00",
      etaLabel: ""
    });

    expect(
      buildIndexProgressSummary({
        status: "ready",
        current: 7062,
        total: 7062,
        message: "The folder index is ready."
      })
    ).toEqual({
      countLabel: "7062 / 7062",
      percent: 100,
      showProgress: false,
      indeterminate: false,
      elapsedLabel: "",
      remainingLabel: "",
      etaLabel: ""
    });
  });

  it("treats the initial folder scan as an indeterminate indexing phase", () => {
    expect(
      buildIndexProgressSummary({
        status: "running",
        phase: "scan",
        current: 512,
        total: 0,
        elapsedSeconds: 7,
        remainingSeconds: null,
        message: "Scanning the selected folder for index updates. Checked 512 files."
      })
    ).toEqual({
      countLabel: "512 files checked",
      percent: 0,
      showProgress: true,
      indeterminate: true,
      elapsedLabel: "Elapsed 00:07",
      remainingLabel: "",
      etaLabel: ""
    });
  });

  it("hides the Stage 2 progress shell again after adaptation finishes", () => {
    expect(
      buildStage2ProgressSummary({
        status: "ready",
        phase: "finalize",
        phaseCurrent: 1,
        phaseTotal: 1,
        message: "Stage 2 weights are ready."
      })
    ).toEqual({
      countLabel: "100%",
      percent: 100,
      detailLabel: "Stage 2 weights are ready.",
      showProgress: false,
      showSteps: false,
      steps: [
        { key: "prepare", label: "Private data", status: "done", percent: 100 },
        { key: "adapt", label: "Model training", status: "done", percent: 100 },
        { key: "validate", label: "Validation", status: "done", percent: 100 },
        { key: "reindex", label: "Rebuild index", status: "done", percent: 100 },
        { key: "finalize", label: "Finalize", status: "done", percent: 100 }
      ],
      elapsedLabel: "",
      remainingLabel: "",
      etaLabel: ""
    });
  });

  it("builds stage 2 progress labels from structured training updates", () => {
    expect(
      buildStage2ProgressSummary({
        status: "running",
        phase: "adapt",
        phaseCurrent: 6,
        phaseTotal: 18,
        current: 8,
        total: 20,
        startedAtMs: Date.parse("2026-04-13T15:00:00+08:00"),
        elapsedSeconds: 120,
        remainingSeconds: 180,
        message: "Training epoch 1 of 1 · step 6 of 18"
      })
    ).toEqual({
      countLabel: "33%",
      percent: 33,
      detailLabel: "Training epoch 1 of 1 · step 6 of 18",
      showProgress: true,
      showSteps: true,
      steps: [
        { key: "prepare", label: "Private data", status: "done", percent: 100 },
        { key: "adapt", label: "Model training", status: "running", percent: 33 },
        { key: "validate", label: "Validation", status: "waiting", percent: 0 },
        { key: "reindex", label: "Rebuild index", status: "waiting", percent: 0 },
        { key: "finalize", label: "Finalize", status: "waiting", percent: 0 }
      ],
      elapsedLabel: "Elapsed 02:00",
      remainingLabel: "Left 03:00",
      etaLabel: "Done by 15:05"
    });
  });

  it("does not report 100 percent when private adaptation data is merely prepared", () => {
    expect(
      buildStage2ProgressSummary({
        status: "running",
        phase: "prepare",
        phaseCurrent: 1,
        phaseTotal: 1,
        elapsedSeconds: 18,
        remainingSeconds: 132,
        message: "Private adaptation data is ready."
      })
    ).toEqual({
      countLabel: "15%",
      percent: 15,
      detailLabel: "Private adaptation data is ready.",
      showProgress: true,
      showSteps: true,
      steps: [
        { key: "prepare", label: "Private data", status: "done", percent: 100 },
        { key: "adapt", label: "Model training", status: "waiting", percent: 0 },
        { key: "validate", label: "Validation", status: "waiting", percent: 0 },
        { key: "reindex", label: "Rebuild index", status: "waiting", percent: 0 },
        { key: "finalize", label: "Finalize", status: "waiting", percent: 0 }
      ],
      elapsedLabel: "Elapsed 00:18",
      remainingLabel: "Left 02:12",
      etaLabel: ""
    });
  });

  it("surfaces the reindex phase as a separate running step", () => {
    expect(
      buildStage2ProgressSummary({
        status: "running",
        phase: "reindex",
        phaseCurrent: 42,
        phaseTotal: 7062,
        elapsedSeconds: 90,
        remainingSeconds: 240,
        message: "Rebuilding the active folder index with Stage 2."
      })
    ).toEqual({
      countLabel: "80%",
      percent: 80,
      detailLabel: "Rebuilding the active folder index with Stage 2.",
      showProgress: true,
      showSteps: true,
      steps: [
        { key: "prepare", label: "Private data", status: "done", percent: 100 },
        { key: "adapt", label: "Model training", status: "done", percent: 100 },
        { key: "validate", label: "Validation", status: "done", percent: 100 },
        { key: "reindex", label: "Rebuild index", status: "running", percent: 1 },
        { key: "finalize", label: "Finalize", status: "waiting", percent: 0 }
      ],
      elapsedLabel: "Elapsed 01:30",
      remainingLabel: "Left 04:00",
      etaLabel: ""
    });
  });

  it("collapses the stage 2 detail steps while the feature is idle", () => {
    expect(
      buildStage2ProgressSummary({
        status: "idle",
        phase: "idle",
        message: "Stage 2 adaptation is idle."
      })
    ).toEqual({
      countLabel: "0%",
      percent: 0,
      detailLabel: "Stage 2 adaptation is idle.",
      showProgress: false,
      showSteps: false,
      steps: [
        { key: "prepare", label: "Private data", status: "waiting", percent: 0 },
        { key: "adapt", label: "Model training", status: "waiting", percent: 0 },
        { key: "validate", label: "Validation", status: "waiting", percent: 0 },
        { key: "reindex", label: "Rebuild index", status: "waiting", percent: 0 },
        { key: "finalize", label: "Finalize", status: "waiting", percent: 0 }
      ],
      elapsedLabel: "",
      remainingLabel: "",
      etaLabel: ""
    });
  });
});
