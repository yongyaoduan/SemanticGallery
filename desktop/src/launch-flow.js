export function createInitialLaunchState() {
  return {
    onboardingStep: "intro"
  };
}

export function resolveOnboardingStep({ onboardingRequired, setupStatus, previousStep = "intro" }) {
  if (setupStatus === "ready" && (onboardingRequired || previousStep === "install")) {
    return "complete";
  }

  if (
    previousStep === "install" &&
    (setupStatus === "running" || setupStatus === "failed" || setupStatus === "cancelling")
  ) {
    return "install";
  }

  if (!onboardingRequired) {
    return null;
  }

  return "intro";
}

export function resolveVisibleScreen({ onboardingRequired, onboardingStep, settingsOpen }) {
  if (onboardingRequired && onboardingStep) {
    return "launch";
  }

  if (settingsOpen) {
    return "settings";
  }

  return "workspace";
}
