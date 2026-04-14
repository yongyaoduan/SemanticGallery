import { describe, expect, it } from "vitest";

import { resolveStage2LaunchGuard } from "./stage2-guard";

describe("stage2 guard", () => {
  it("rejects folders with fewer than 100 indexed images", () => {
    expect(
      resolveStage2LaunchGuard({
        setupStatus: "ready",
        activeFolder: "/tmp/gallery",
        indexingStatus: "ready",
        indexedImageCount: 99,
        stage2Status: "idle"
      })
    ).toBe(
      "This folder currently has 99 supported images. Stage 2 adaptation is not necessary yet. Add at least 100 supported images to the active folder before running it."
    );
  });

  it("allows Stage 2 once the active folder has at least 100 indexed images", () => {
    expect(
      resolveStage2LaunchGuard({
        setupStatus: "ready",
        activeFolder: "/tmp/gallery",
        indexingStatus: "ready",
        indexedImageCount: 100,
        stage2Status: "idle"
      })
    ).toBeNull();
  });
});
