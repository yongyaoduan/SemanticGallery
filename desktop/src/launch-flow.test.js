import { describe, expect, it } from "vitest";

import {
  createInitialLaunchState,
  resolveOnboardingStep,
  resolveVisibleScreen
} from "./launch-flow";

describe("launch flow", () => {
  it("starts from the intro page until the runtime reports its real onboarding state", () => {
    expect(createInitialLaunchState()).toEqual({
      onboardingStep: "intro"
    });
  });

  it("keeps first launch on the intro page until installation starts", () => {
    expect(
      resolveOnboardingStep({
        onboardingRequired: true,
        setupStatus: "idle"
      })
    ).toBe("intro");
  });

  it("keeps the install page visible while setup is running or failed", () => {
    expect(
      resolveOnboardingStep({
        onboardingRequired: true,
        setupStatus: "running",
        previousStep: "install"
      })
    ).toBe("install");

    expect(
      resolveOnboardingStep({
        onboardingRequired: true,
        setupStatus: "failed",
        previousStep: "install"
      })
    ).toBe("install");
  });

  it("switches to the completion page when setup is ready for a first-launch user", () => {
    expect(
      resolveOnboardingStep({
        onboardingRequired: true,
        setupStatus: "ready",
        previousStep: "install"
      })
    ).toBe("complete");
  });

  it("keeps the completion page after install even if an old onboarding marker already exists", () => {
    expect(
      resolveOnboardingStep({
        onboardingRequired: false,
        setupStatus: "ready",
        previousStep: "install"
      })
    ).toBe("complete");
  });

  it("skips onboarding screens after the first-time flow is completed", () => {
    expect(
      resolveOnboardingStep({
        onboardingRequired: false,
        setupStatus: "ready",
        previousStep: "complete"
      })
    ).toBeNull();

    expect(
      resolveVisibleScreen({
        onboardingRequired: false,
        onboardingStep: null,
        settingsOpen: false
      })
    ).toBe("workspace");
  });

  it("keeps the launch screen visible whenever onboarding is still required", () => {
    expect(
      resolveVisibleScreen({
        onboardingRequired: true,
        onboardingStep: "intro",
        settingsOpen: true
      })
    ).toBe("launch");
  });
});
