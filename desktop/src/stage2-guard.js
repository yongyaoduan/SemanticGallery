export function resolveStage2LaunchGuard({
  setupStatus,
  activeFolder,
  indexingStatus,
  indexedImageCount,
  stage2Status
}) {
  if (setupStatus !== "ready") {
    return "Install the local runtime before running Stage 2 adaptation.";
  }
  if (!activeFolder) {
    return "Choose a folder before running Stage 2 adaptation.";
  }
  if (indexingStatus === "running") {
    return "Wait for the current folder index to finish before running Stage 2 adaptation.";
  }
  if (stage2Status === "running") {
    return "Stage 2 adaptation is already running.";
  }
  const supportedImageCount = Number(indexedImageCount) || 0;
  if (supportedImageCount < 100) {
    return `This folder currently has ${supportedImageCount} supported images. Stage 2 adaptation is not necessary yet. Add at least 100 supported images to the active folder before running it.`;
  }
  return null;
}
