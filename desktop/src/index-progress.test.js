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
      elapsedLabel: "Elapsed 05:15",
      remainingLabel: "",
      etaLabel: ""
    });
  });

  it("builds stage 2 progress labels from structured training updates", () => {
    expect(
      buildStage2ProgressSummary({
        status: "running",
        phase: "adapt",
        current: 8,
        total: 20,
        startedAtMs: Date.parse("2026-04-13T15:00:00+08:00"),
        elapsedSeconds: 120,
        remainingSeconds: 180,
        message: "Training epoch 1 of 1 · step 6 of 18"
      })
    ).toEqual({
      countLabel: "40%",
      percent: 40,
      detailLabel: "Training epoch 1 of 1 · step 6 of 18",
      elapsedLabel: "Elapsed 02:00",
      remainingLabel: "Left 03:00",
      etaLabel: "Done by 15:05"
    });
  });
});
