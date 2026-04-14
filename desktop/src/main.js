import "./styles.css";

import { listen } from "@tauri-apps/api/event";

import {
  cancelRuntimeSetup,
  completeOnboarding,
  desktopAutomationToolsEnabled,
  desktopDeleteImage,
  desktopDeleteImages,
  desktopMetadata,
  desktopRefreshFolder,
  desktopRunStage2,
  desktopRuntimeStatus,
  desktopSearchSimilar,
  desktopSearchText,
  desktopSearchUploadedImage,
  desktopSelectFolder,
  openUninstaller,
  pickFolder,
  resetSidecarBaseUrl,
  sidecarBaseUrl,
  toAssetUrl,
  runtimeBootstrapState,
  startRuntime
} from "./api-client";
import { createInitialLaunchState, resolveOnboardingStep, resolveVisibleScreen } from "./launch-flow";
import { createInitialState, reduceAction } from "./state";
import {
  countVisibleSelection,
  pruneSelectionToVisibleResults,
  toggleAllVisibleResults,
  toggleResultSelection
} from "./workspace-selection";
import { extractImageFileFromClipboardItems } from "./clipboard-image";
import { buildIndexProgressSummary, buildStage2ProgressSummary } from "./index-progress";
import { attachImageFallback, buildImageSourceCandidates } from "./image-fallback";
import { buildQueryPresentation } from "./query-presentation";
import { nextSetupLogPinState, shouldAutoScrollSetupLogs } from "./setup-log-scroll";
import { resolveStage2LaunchGuard } from "./stage2-guard";
import { buildTrashConfirmationContent } from "./trash-confirmation";

const SETUP_PROGRESS_EVENT = "semanticgallery://setup-progress";
const SETUP_LOG_EVENT = "semanticgallery://setup-log";
const SETUP_ERROR_EVENT = "semanticgallery://setup-error";
const RUNTIME_READY_EVENT = "semanticgallery://runtime-ready";
const INDEX_PROGRESS_EVENT = "semanticgallery://folder-index-progress";
const DEFAULT_LIMIT = 25;
const AUTO_REFRESH_INTERVAL_MS = 5000;

function createEmptyQuery() {
  return {
    kind: "none",
    text: "",
    url: "",
    relativePath: "",
    fileName: "",
    file: null,
    previewUrl: "",
    previewUrlOwned: false
  };
}

let state = createInitialState();
let bootstrapPollHandle = null;
let runtimeStatusLoaded = false;
let runtimeStatusPromise = null;
let runtimeEventSource = null;
let onboardingStep = createInitialLaunchState().onboardingStep;
let selectionMode = false;
let selectedPaths = new Set();
let activeMetadataController = null;
let lightboxItem = null;
let currentQuery = createEmptyQuery();
let pendingTrashConfirmation = null;
let setupLogsPinnedToBottom = true;
let autoRefreshHandle = null;
let automationToolsEnabled = false;

const elements = {
  launchScreen: document.getElementById("launch-screen"),
  workspaceScreen: document.getElementById("workspace-screen"),
  settingsScreen: document.getElementById("settings-screen"),
  launchIntro: document.getElementById("launch-intro"),
  launchProgressShell: document.getElementById("launch-progress-shell"),
  launchComplete: document.getElementById("launch-complete"),
  setupTitle: document.getElementById("setup-title"),
  setupMessage: document.getElementById("setup-message"),
  setupProgressLabel: document.getElementById("setup-progress-label"),
  setupProgressBar: document.getElementById("setup-progress-bar"),
  setupStepList: document.getElementById("setup-step-list"),
  setupStepStatus: document.getElementById("setup-step-status"),
  setupLogCount: document.getElementById("setup-log-count"),
  setupLogList: document.getElementById("setup-log-list"),
  startSetupButton: document.getElementById("start-setup-button"),
  cancelSetupButton: document.getElementById("cancel-setup-button"),
  finishSetupButton: document.getElementById("finish-setup-button"),
  activeFolderLabel: document.getElementById("active-folder-label"),
  encoderStatusLabel: document.getElementById("encoder-status-label"),
  refreshStatusLabel: document.getElementById("refresh-status-label"),
  settingsFolderValue: document.getElementById("settings-folder-value"),
  settingsIndexProgressShell: document.getElementById("settings-index-progress-shell"),
  settingsIndexProgressLabel: document.getElementById("settings-index-progress-label"),
  settingsIndexProgressBar: document.getElementById("settings-index-progress-bar"),
  settingsIndexProgressTimeRow: document.getElementById("settings-index-progress-time-row"),
  settingsIndexProgressElapsed: document.getElementById("settings-index-progress-elapsed"),
  settingsIndexProgressRemaining: document.getElementById("settings-index-progress-remaining"),
  settingsIndexProgressEta: document.getElementById("settings-index-progress-eta"),
  settingsIndexMessage: document.getElementById("settings-index-message"),
  settingsStage2Feedback: document.getElementById("settings-stage2-feedback"),
  settingsStage2ProgressShell: document.getElementById("settings-stage2-progress-shell"),
  settingsStage2ProgressLabel: document.getElementById("settings-stage2-progress-label"),
  settingsStage2ProgressBar: document.getElementById("settings-stage2-progress-bar"),
  settingsStage2ProgressTimeRow: document.getElementById("settings-stage2-progress-time-row"),
  settingsStage2ProgressElapsed: document.getElementById("settings-stage2-progress-elapsed"),
  settingsStage2ProgressRemaining: document.getElementById("settings-stage2-progress-remaining"),
  settingsStage2ProgressEta: document.getElementById("settings-stage2-progress-eta"),
  settingsStage2Message: document.getElementById("settings-stage2-message"),
  settingsStage2StepList: document.getElementById("settings-stage2-step-list"),
  lastTaskMessage: document.getElementById("last-task-message"),
  openUninstallerButton: document.getElementById("open-uninstaller-button"),
  resultsCountLabel: document.getElementById("results-count-label"),
  results: document.getElementById("results"),
  resultCardTemplate: document.getElementById("result-card-template"),
  resultLimit: document.getElementById("result-limit"),
  selectionToggle: document.getElementById("selection-toggle"),
  selectionBar: document.getElementById("selection-bar"),
  selectionCount: document.getElementById("selection-count"),
  selectionSelectAll: document.getElementById("selection-select-all"),
  selectionClear: document.getElementById("selection-clear"),
  selectionDelete: document.getElementById("selection-delete"),
  refreshButton: document.getElementById("refresh-button"),
  settingsButton: document.getElementById("settings-button"),
  settingsBackButton: document.getElementById("settings-back-button"),
  chooseFolderButton: document.getElementById("choose-folder-button"),
  automationFolderShell: document.getElementById("automation-folder-shell"),
  automationFolderInput: document.getElementById("automation-folder-input"),
  automationFolderSubmit: document.getElementById("automation-folder-submit"),
  runStage2Button: document.getElementById("run-stage2-button"),
  searchForm: document.getElementById("search-form"),
  searchInput: document.getElementById("search-input"),
  automationImageShell: document.getElementById("automation-image-shell"),
  automationImageInput: document.getElementById("automation-image-input"),
  automationImageSubmit: document.getElementById("automation-image-submit"),
  searchQueryChip: document.getElementById("search-query-chip"),
  searchQueryChipImage: document.getElementById("search-query-chip-image"),
  searchQueryChipLabel: document.getElementById("search-query-chip-label"),
  clearQueryChipButton: document.getElementById("clear-query-chip-button"),
  lightbox: document.getElementById("lightbox"),
  lightboxImage: document.getElementById("lightbox-image"),
  lightboxClose: document.getElementById("lightbox-close"),
  lightboxInfoToggle: document.getElementById("lightbox-info-toggle"),
  lightboxDelete: document.getElementById("lightbox-delete"),
  lightboxSimilar: document.getElementById("lightbox-similar"),
  lightboxMeta: document.getElementById("lightbox-meta"),
  lightboxFilename: document.getElementById("lightbox-filename"),
  lightboxPath: document.getElementById("lightbox-path"),
  lightboxSize: document.getElementById("lightbox-size"),
  lightboxDimensions: document.getElementById("lightbox-dimensions"),
  lightboxTimeLabel: document.getElementById("lightbox-time-label"),
  lightboxTime: document.getElementById("lightbox-time"),
  confirmDialog: document.getElementById("confirm-dialog"),
  confirmDialogTitle: document.getElementById("confirm-dialog-title"),
  confirmDialogMessage: document.getElementById("confirm-dialog-message"),
  confirmDialogCancel: document.getElementById("confirm-dialog-cancel"),
  confirmDialogAccept: document.getElementById("confirm-dialog-accept")
};

function dispatch(action) {
  state = reduceAction(state, action);
  onboardingStep = resolveOnboardingStep({
    onboardingRequired: state.setup.onboardingRequired,
    setupStatus: state.setup.status,
    previousStep: onboardingStep
  });
  render();
}

function setBootstrapPolling(active) {
  if (bootstrapPollHandle) {
    clearInterval(bootstrapPollHandle);
    bootstrapPollHandle = null;
  }
  if (!active) {
    return;
  }
  bootstrapPollHandle = window.setInterval(() => {
    syncBootstrapState().catch(() => {
      // The failure state is already reduced into UI state.
    });
  }, 900);
}

function normalizeLimit() {
  const raw = Number.parseInt(elements.resultLimit.value, 10);
  if (Number.isNaN(raw) || raw < 1) {
    return DEFAULT_LIMIT;
  }
  return Math.min(raw, 100);
}

function indexedImageCount() {
  return Math.max(Number(state.indexing.current) || 0, Number(state.indexing.total) || 0);
}

function revokeOwnedQueryPreview(query) {
  if (query?.previewUrlOwned && query.previewUrl?.startsWith("blob:")) {
    URL.revokeObjectURL(query.previewUrl);
  }
}

function replaceCurrentQuery(nextQuery) {
  revokeOwnedQueryPreview(currentQuery);
  currentQuery = {
    ...createEmptyQuery(),
    ...nextQuery
  };
}

function clearCurrentQuery({ clearResults = false } = {}) {
  replaceCurrentQuery(createEmptyQuery());
  elements.searchInput.value = "";
  if (clearResults) {
    dispatch({ type: "results-received", payload: { results: [] } });
  }
}

async function loadAutomationToolsStatus() {
  try {
    automationToolsEnabled = automationToolsEnabled || Boolean(await desktopAutomationToolsEnabled());
  } catch {
    // Keep the previous value when the bridge is not ready yet.
  }
}

function normalizeResultsPayload(payload) {
  return {
    ...payload,
    results: (payload.results ?? []).map((item) => ({
      ...item,
      thumbnailUrl: toAssetUrl(item.thumbnailPath ?? item.thumbnailUrl),
      fullUrl: toAssetUrl(item.fullPath ?? item.fullUrl)
    }))
  };
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const payload = typeof reader.result === "string" ? reader.result.split(",").at(-1) : "";
      if (!payload) {
        reject(new Error("Unsupported image payload."));
        return;
      }
      resolve(payload);
    };
    reader.onerror = () => {
      reject(new Error("Unsupported image payload."));
    };
    reader.readAsDataURL(blob);
  });
}

async function syncBootstrapState() {
  const payload = await runtimeBootstrapState();
  dispatch({ type: "bootstrap-snapshot", payload });
  if (payload.status === "ready") {
    setBootstrapPolling(false);
    await handleRuntimeReady();
  } else if (payload.status === "failed" || payload.status === "idle") {
    setBootstrapPolling(false);
  }
}

async function syncRuntimeStatusSnapshot() {
  const payload = await desktopRuntimeStatus();
  dispatch({ type: "runtime-status", payload });
  return payload;
}

async function handleRuntimeReady() {
  await loadAutomationToolsStatus();
  if (runtimeStatusLoaded) {
    await connectRuntimeEvents();
    return;
  }
  if (runtimeStatusPromise) {
    await runtimeStatusPromise;
    return;
  }
  runtimeStatusPromise = loadRuntimeStatus()
    .then(() => {
      runtimeStatusLoaded = true;
    })
    .finally(() => {
      runtimeStatusPromise = null;
    });
  await runtimeStatusPromise;
  await connectRuntimeEvents();
}

async function connectRuntimeEvents() {
  if (runtimeEventSource) {
    return;
  }

  const baseUrl = await sidecarBaseUrl();
  runtimeEventSource = new EventSource(`${baseUrl}/api/events`);
  runtimeEventSource.addEventListener("stage2-progress", (event) => {
    dispatch({ type: "stage2-progress", payload: JSON.parse(event.data) });
  });
}

async function triggerRuntimeStart({ force = false } = {}) {
  if (force) {
    resetSidecarBaseUrl();
    runtimeStatusLoaded = false;
  }

  try {
    const payload = await startRuntime();
    dispatch({ type: "bootstrap-snapshot", payload });
    if (payload.status === "ready") {
      await handleRuntimeReady();
      return;
    }
    if (payload.status === "running") {
      setBootstrapPolling(true);
    }
  } catch (error) {
    dispatch({ type: "setup-failed", payload: error.message });
  }
}

function formatEncoderLabel(signature) {
  return signature && signature !== "stage1" ? "Stage 2" : "Stage 1";
}

function formatByteSize(byteSize) {
  if (!byteSize) {
    return "Unknown";
  }
  const units = ["B", "KB", "MB", "GB"];
  let value = byteSize;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value >= 10 || unitIndex === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIndex]}`;
}

function createEmptyState(message, actionLabel = null) {
  const empty = document.createElement("div");
  empty.className = "empty-state";

  const copy = document.createElement("p");
  copy.textContent = message;
  empty.append(copy);

  if (actionLabel) {
    const button = document.createElement("button");
    button.className = "secondary-button";
    button.type = "button";
    button.textContent = actionLabel;
    button.addEventListener("click", () => dispatch({ type: "settings-opened" }));
    empty.append(button);
  }

  return empty;
}

function resetSelection({ keepMode = false } = {}) {
  if (!keepMode) {
    selectionMode = false;
  }
  selectedPaths = new Set();
}

function updateSelectionBar(results = state.results) {
  selectedPaths = pruneSelectionToVisibleResults(results, selectedPaths);
  const visibleCount = results.length;
  const selectedCount = countVisibleSelection(results, selectedPaths);
  const allVisibleSelected = visibleCount > 0 && selectedCount === visibleCount;

  elements.selectionBar.hidden = !selectionMode;
  elements.selectionToggle.classList.toggle("is-active", selectionMode);
  elements.selectionCount.textContent = `${selectedCount} selected`;
  elements.selectionDelete.disabled = selectedCount === 0;
  elements.selectionSelectAll.disabled = visibleCount === 0;
  elements.selectionSelectAll.classList.toggle("is-active", allVisibleSelected);
  elements.selectionSelectAll.title = allVisibleSelected ? "Clear visible selection" : "Select all visible results";
  elements.selectionSelectAll.setAttribute(
    "aria-label",
    allVisibleSelected ? "Clear visible selection" : "Select all visible results"
  );
}

function renderSelectionState() {
  for (const card of elements.results.querySelectorAll(".result-card")) {
    const relativePath = card.dataset.relativePath || "";
    card.classList.toggle("is-selectable", selectionMode);
    card.classList.toggle("is-selected", selectedPaths.has(relativePath));
  }
}

function updateBodyLock() {
  const isLocked = !elements.lightbox.hidden || !elements.confirmDialog.hidden;
  document.body.classList.toggle("is-locked", isLocked);
}

function settleTrashConfirmation(confirmed) {
  if (!pendingTrashConfirmation) {
    elements.confirmDialog.hidden = true;
    elements.confirmDialog.style.display = "none";
    updateBodyLock();
    return;
  }

  const { resolve } = pendingTrashConfirmation;
  pendingTrashConfirmation = null;
  elements.confirmDialog.hidden = true;
  elements.confirmDialog.style.display = "none";
  updateBodyLock();
  resolve(confirmed);
}

function requestTrashConfirmation(options) {
  if (pendingTrashConfirmation) {
    pendingTrashConfirmation.resolve(false);
    pendingTrashConfirmation = null;
  }

  const content = buildTrashConfirmationContent(options);
  elements.confirmDialogTitle.textContent = content.title;
  elements.confirmDialogMessage.textContent = content.message;
  elements.confirmDialogAccept.textContent = content.confirmLabel;
  elements.confirmDialog.hidden = false;
  elements.confirmDialog.style.display = "";
  updateBodyLock();

  return new Promise((resolve) => {
    pendingTrashConfirmation = { resolve };
    window.setTimeout(() => {
      elements.confirmDialogAccept.focus();
    }, 0);
  });
}

function closeLightbox() {
  lightboxItem = null;
  elements.lightbox.hidden = true;
  elements.lightboxImage.removeAttribute("src");
  elements.lightboxImage.removeAttribute("alt");
  elements.lightboxMeta.hidden = true;
  elements.lightboxInfoToggle.setAttribute("aria-expanded", "false");
  elements.lightboxFilename.textContent = "";
  elements.lightboxPath.textContent = "";
  elements.lightboxSize.textContent = "";
  elements.lightboxDimensions.textContent = "";
  elements.lightboxTimeLabel.textContent = "Time";
  elements.lightboxTime.textContent = "";
  elements.lightboxDelete.disabled = false;
  elements.lightboxSimilar.disabled = true;
  if (activeMetadataController) {
    activeMetadataController.abort();
    activeMetadataController = null;
  }
  updateBodyLock();
}

async function openLightbox(item) {
  lightboxItem = item;
  attachImageFallback(
    elements.lightboxImage,
    buildImageSourceCandidates(item.fullUrl, item.thumbnailUrl)
  );
  elements.lightboxImage.alt = item.name ?? item.fileName ?? "Image";
  elements.lightboxMeta.hidden = true;
  elements.lightboxInfoToggle.setAttribute("aria-expanded", "false");
  elements.lightboxFilename.textContent = item.fileName ?? "";
  elements.lightboxPath.textContent = "";
  elements.lightboxSize.textContent = "Loading";
  elements.lightboxDimensions.textContent = "Loading";
  elements.lightboxTimeLabel.textContent = "Time";
  elements.lightboxTime.textContent = "Loading";
  elements.lightboxDelete.disabled = !item.relativePath;
  elements.lightboxSimilar.disabled = !item.relativePath;
  elements.lightbox.hidden = false;
  updateBodyLock();

  if (activeMetadataController) {
    activeMetadataController.abort();
  }
  activeMetadataController = new AbortController();

  try {
    const metadata = await desktopMetadata(item.relativePath ?? "");
    if (lightboxItem?.relativePath !== item.relativePath) {
      return;
    }
    elements.lightboxFilename.textContent = metadata.fileName ?? item.fileName ?? "";
    elements.lightboxPath.textContent = metadata.path ?? "";
    elements.lightboxSize.textContent = formatByteSize(metadata.byteSize);
    elements.lightboxDimensions.textContent =
      metadata.width && metadata.height ? `${metadata.width} x ${metadata.height}` : "Unknown";
    elements.lightboxTimeLabel.textContent = metadata.timeLabel ?? "Time";
    elements.lightboxTime.textContent = metadata.timeValue ?? "Unknown";
  } catch (error) {
    if (error.name !== "AbortError") {
      elements.lightboxPath.textContent = "Unknown";
      elements.lightboxSize.textContent = "Unknown";
      elements.lightboxDimensions.textContent = "Unknown";
      elements.lightboxTimeLabel.textContent = "Time";
      elements.lightboxTime.textContent = "Unknown";
    }
  }
}

function renderResults(results) {
  elements.results.replaceChildren();
  elements.resultsCountLabel.textContent = `${results.length} ${results.length === 1 ? "item" : "items"}`;

  selectedPaths = pruneSelectionToVisibleResults(results, selectedPaths);
  if (!results.length) {
    if (!state.activeFolder) {
      elements.results.append(createEmptyState("Choose a folder to start.", "Open Settings"));
    } else if (currentQuery.kind === "similar") {
      elements.results.append(createEmptyState("No similar images matched the current folder yet."));
    } else {
      elements.results.append(createEmptyState("No matches yet. Try another description or refresh the folder index."));
    }
    updateSelectionBar(results);
    renderSelectionState();
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const item of results) {
    const node = elements.resultCardTemplate.content.firstElementChild.cloneNode(true);
    node.dataset.relativePath = item.relativePath ?? "";
    node.setAttribute("aria-label", item.fileName ?? item.name ?? "Image");
    const image = node.querySelector(".result-thumb");
    image.alt = item.name ?? item.fileName ?? "Image";
    attachImageFallback(image, buildImageSourceCandidates(item.thumbnailUrl, item.fullUrl));
    fragment.append(node);
  }
  elements.results.append(fragment);
  updateSelectionBar(results);
  renderSelectionState();
}

function renderSetupSteps() {
  elements.setupStepList.replaceChildren();
  const fragment = document.createDocumentFragment();

  for (const step of state.setup.steps) {
    const row = document.createElement("div");
    row.className = `setup-step is-${step.status}`;

    const meta = document.createElement("div");
    meta.className = "setup-step-meta";
    const label = document.createElement("span");
    label.textContent = step.label;
    const status = document.createElement("span");
    status.textContent =
      step.status === "done" ? "Ready" : step.status === "running" ? "Working" : "Waiting";
    meta.append(label, status);

    const bar = document.createElement("div");
    bar.className = "setup-step-track";
    const fill = document.createElement("span");
    fill.style.width = step.status === "done" ? "100%" : step.status === "running" ? "64%" : "0%";
    bar.append(fill);

    row.append(meta, bar);
    fragment.append(row);
  }

  elements.setupStepList.append(fragment);
}

function renderStage2Steps(steps) {
  elements.settingsStage2StepList.replaceChildren();
  const fragment = document.createDocumentFragment();

  for (const step of steps) {
    const row = document.createElement("div");
    row.className = `setup-step is-${step.status}`;

    const meta = document.createElement("div");
    meta.className = "setup-step-meta";
    const label = document.createElement("span");
    label.textContent = step.label;
    const status = document.createElement("span");
    status.textContent =
      step.status === "done"
        ? "Ready"
        : step.status === "failed"
          ? "Failed"
          : step.status === "running"
            ? "Working"
            : "Waiting";
    meta.append(label, status);

    const bar = document.createElement("div");
    bar.className = "setup-step-track";
    const fill = document.createElement("span");
    const runningPercent = step.percent > 0 ? step.percent : 64;
    fill.style.width =
      step.status === "done"
        ? "100%"
        : step.status === "failed"
          ? `${Math.max(step.percent, 22)}%`
          : step.status === "running"
            ? `${runningPercent}%`
            : "0%";
    bar.append(fill);

    row.append(meta, bar);
    fragment.append(row);
  }

  elements.settingsStage2StepList.append(fragment);
}

function renderSetupLogs() {
  const shouldStickToBottom = shouldAutoScrollSetupLogs({
    pinnedToBottom: setupLogsPinnedToBottom,
    scrollTop: elements.setupLogList.scrollTop,
    clientHeight: elements.setupLogList.clientHeight,
    scrollHeight: elements.setupLogList.scrollHeight
  });
  elements.setupLogList.replaceChildren();
  elements.setupLogCount.textContent = `${state.setup.logs.length} lines`;

  if (!state.setup.logs.length) {
    const empty = document.createElement("p");
    empty.className = "setup-log-empty";
    empty.textContent = "No log lines yet.";
    elements.setupLogList.append(empty);
    if (shouldStickToBottom) {
      elements.setupLogList.scrollTop = elements.setupLogList.scrollHeight;
      setupLogsPinnedToBottom = true;
    }
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const line of state.setup.logs) {
    const row = document.createElement("p");
    row.className = "setup-log-line";
    row.textContent = line;
    fragment.append(row);
  }
  elements.setupLogList.append(fragment);
  if (shouldStickToBottom) {
    elements.setupLogList.scrollTop = elements.setupLogList.scrollHeight;
    setupLogsPinnedToBottom = true;
  }
}

function render() {
  const setupPercent =
    state.setup.totalSteps > 0 ? Math.round((state.setup.currentStep / state.setup.totalSteps) * 100) : 0;
  const visibleScreen = resolveVisibleScreen({
    onboardingRequired: state.setup.onboardingRequired,
    onboardingStep,
    settingsOpen: state.settingsOpen
  });
  const launchVisible = visibleScreen === "launch";
  const settingsVisible = visibleScreen === "settings";
  const workspaceVisible = visibleScreen === "workspace";
  const setupRunning = state.setup.status === "running" || state.setup.status === "cancelling";
  const showLaunchIntro = onboardingStep === "intro";
  const showLaunchProgress = onboardingStep === "install";
  const showLaunchComplete = onboardingStep === "complete";
  const indexSummary = buildIndexProgressSummary(state.indexing);
  const stage2Summary = buildStage2ProgressSummary(state.stage2);

  elements.launchScreen.hidden = !launchVisible;
  elements.workspaceScreen.hidden = !workspaceVisible;
  elements.settingsScreen.hidden = !settingsVisible;
  elements.launchScreen.style.display = launchVisible ? "" : "none";
  elements.workspaceScreen.style.display = workspaceVisible ? "" : "none";
  elements.settingsScreen.style.display = settingsVisible ? "" : "none";
  elements.launchIntro.style.display = showLaunchIntro ? "grid" : "none";
  elements.launchProgressShell.style.display = showLaunchProgress ? "grid" : "none";
  elements.launchComplete.style.display = showLaunchComplete ? "grid" : "none";

  if (showLaunchComplete) {
    elements.setupTitle.textContent = "Desktop runtime is ready";
  } else if (state.setup.status === "failed") {
    elements.setupTitle.textContent = "Desktop runtime needs attention";
  } else if (state.setup.status === "cancelling") {
    elements.setupTitle.textContent = "Stopping the local runtime install";
  } else if (state.setup.status === "running") {
    elements.setupTitle.textContent = "Installing your local search engine";
  } else {
    elements.setupTitle.textContent = "Install the local runtime";
  }

  elements.setupMessage.textContent = state.setup.message;
  elements.setupProgressLabel.textContent = `${state.setup.currentStep} / ${state.setup.totalSteps}`;
  elements.setupProgressBar.style.width = `${setupPercent}%`;
  elements.setupStepStatus.textContent =
    state.setup.status === "failed"
      ? "Needs attention"
      : state.setup.status === "cancelling"
        ? "Stopping"
        : state.setup.status === "ready"
          ? "Ready"
          : "Working";
  elements.startSetupButton.hidden = !showLaunchIntro && state.setup.status !== "failed";
  elements.startSetupButton.style.display =
    !showLaunchIntro && state.setup.status !== "failed" ? "none" : "inline-flex";
  elements.startSetupButton.textContent = state.setup.status === "failed" ? "Try Again" : "Install And Start";
  elements.cancelSetupButton.hidden = !setupRunning;
  elements.cancelSetupButton.style.display = setupRunning ? "inline-flex" : "none";
  elements.cancelSetupButton.disabled = state.setup.status === "cancelling";

  elements.activeFolderLabel.textContent = state.activeFolder ?? "No folder selected";
  elements.settingsFolderValue.textContent = state.activeFolder ?? "No folder selected";
  elements.encoderStatusLabel.textContent = formatEncoderLabel(state.activeEncoderSignature);
  elements.refreshStatusLabel.textContent =
    state.indexing.status === "running" ? "Indexing" : state.refresh.label;
  elements.lastTaskMessage.textContent = state.lastTaskMessage || "";
  elements.settingsIndexProgressShell.hidden = !indexSummary.showProgress;
  elements.settingsIndexProgressShell.style.display = indexSummary.showProgress ? "grid" : "none";
  elements.settingsIndexProgressLabel.textContent = indexSummary.countLabel;
  elements.settingsIndexProgressBar.parentElement.classList.toggle("is-indeterminate", indexSummary.indeterminate);
  elements.settingsIndexProgressBar.style.width = indexSummary.indeterminate ? "34%" : `${indexSummary.percent}%`;
  const showProgressTiming = Boolean(indexSummary.elapsedLabel || indexSummary.remainingLabel || indexSummary.etaLabel);
  elements.settingsIndexProgressTimeRow.hidden = !showProgressTiming;
  elements.settingsIndexProgressTimeRow.style.display = showProgressTiming ? "flex" : "none";
  elements.settingsIndexProgressElapsed.textContent = indexSummary.elapsedLabel;
  elements.settingsIndexProgressRemaining.textContent = indexSummary.remainingLabel;
  elements.settingsIndexProgressEta.textContent = indexSummary.etaLabel;
  elements.settingsIndexMessage.textContent = state.indexing.message;
  elements.settingsStage2ProgressShell.hidden = !stage2Summary.showProgress;
  elements.settingsStage2ProgressShell.style.display = stage2Summary.showProgress ? "grid" : "none";
  elements.settingsStage2ProgressLabel.textContent = stage2Summary.countLabel;
  elements.settingsStage2ProgressBar.parentElement.classList.remove("is-indeterminate");
  elements.settingsStage2ProgressBar.style.width = `${stage2Summary.percent}%`;
  const showStage2Timing = Boolean(stage2Summary.elapsedLabel || stage2Summary.remainingLabel || stage2Summary.etaLabel);
  elements.settingsStage2ProgressTimeRow.hidden = !showStage2Timing;
  elements.settingsStage2ProgressTimeRow.style.display = showStage2Timing ? "flex" : "none";
  elements.settingsStage2ProgressElapsed.textContent = stage2Summary.elapsedLabel;
  elements.settingsStage2ProgressRemaining.textContent = stage2Summary.remainingLabel;
  elements.settingsStage2ProgressEta.textContent = stage2Summary.etaLabel;
  const showStage2Feedback = state.stage2.status === "failed" && Boolean(state.stage2.message);
  elements.settingsStage2Feedback.hidden = !showStage2Feedback;
  elements.settingsStage2Feedback.style.display = showStage2Feedback ? "block" : "none";
  elements.settingsStage2Feedback.textContent = showStage2Feedback ? state.stage2.message : "";
  elements.settingsStage2Message.textContent = stage2Summary.detailLabel || state.stage2.message;
  elements.settingsStage2StepList.hidden = !stage2Summary.showSteps;
  elements.settingsStage2StepList.style.display = stage2Summary.showSteps ? "grid" : "none";

  const stage2Running = state.stage2.status === "running";
  elements.refreshButton.disabled = !state.activeFolder || state.refresh.isRunning || stage2Running;
  elements.searchInput.disabled = !state.activeFolder;
  elements.resultLimit.disabled = !state.activeFolder;
  elements.selectionToggle.disabled = !state.results.length;
  elements.chooseFolderButton.disabled =
    state.setup.status !== "ready" || state.indexing.status === "running" || stage2Running;
  elements.runStage2Button.disabled =
    state.setup.status !== "ready" || !state.activeFolder || state.indexing.status === "running" || stage2Running;
  elements.searchInput.placeholder = state.activeFolder
    ? "Describe the photo you want to find"
    : "Choose a folder first";
  elements.automationFolderShell.hidden = !automationToolsEnabled;
  elements.automationFolderShell.style.display = automationToolsEnabled ? "grid" : "none";
  elements.automationFolderSubmit.disabled = state.setup.status !== "ready" || state.indexing.status === "running" || stage2Running;
  elements.automationImageShell.hidden = !automationToolsEnabled;
  elements.automationImageShell.style.display = automationToolsEnabled ? "grid" : "none";
  elements.automationImageSubmit.disabled = !state.activeFolder;
  const queryPresentation = buildQueryPresentation(currentQuery);
  elements.searchForm.classList.toggle("has-query-chip", Boolean(queryPresentation.chip));
  if (currentQuery.kind === "text") {
    elements.searchInput.value = queryPresentation.inputValue;
  } else if (currentQuery.kind !== "none") {
    elements.searchInput.value = "";
  }
  elements.searchQueryChip.hidden = !queryPresentation.chip;
  elements.searchQueryChip.style.display = queryPresentation.chip ? "" : "none";
  if (queryPresentation.chip) {
    elements.searchQueryChipImage.src = queryPresentation.chip.previewUrl;
    elements.searchQueryChipImage.alt = queryPresentation.chip.label;
    elements.searchQueryChipLabel.textContent = queryPresentation.chip.label;
  } else {
    elements.searchQueryChipImage.removeAttribute("src");
    elements.searchQueryChipImage.alt = "";
    elements.searchQueryChipLabel.textContent = "";
  }

  renderSetupSteps();
  if (stage2Summary.showSteps) {
    renderStage2Steps(stage2Summary.steps ?? []);
  } else {
    elements.settingsStage2StepList.replaceChildren();
  }
  renderSetupLogs();
  renderResults(state.results);
}

async function loadRuntimeStatus() {
  try {
    const payload = await desktopRuntimeStatus();
    dispatch({ type: "runtime-status", payload });
    dispatch({ type: "setup-finished" });
  } catch (error) {
    dispatch({ type: "setup-failed", payload: error.message });
    throw error;
  }
}

async function refreshFolder({ lightweight = false, silent = false } = {}) {
  if (!silent) {
    dispatch({ type: "refresh-started" });
    dispatch({
      type: "index-progress",
      payload: {
        status: "running",
        phase: "start",
        current: 0,
        total: 0,
        embeddedCount: 0,
        reusedCount: 0,
        message: "Checking the active folder for index updates."
      }
    });
  }
  try {
    const payload = await desktopRefreshFolder({ lightweight });
    dispatch({ type: "folder-selected", payload });
    if (!silent) {
      dispatch({ type: "refresh-finished", payload: { label: payload.skipped ? "Up to date" : "Refreshed" } });
    }
    if (payload.refreshed && currentQuery.kind !== "none") {
      await rerunCurrentSearch();
    }
  } catch (error) {
    if (!silent) {
      dispatch({ type: "refresh-finished", payload: { label: "Refresh failed" } });
    }
    dispatch({ type: "task-message", payload: error.message });
  }
}

async function selectFolderByPath(folderPath) {
  closeLightbox();
  resetSelection();
  clearCurrentQuery();
  dispatch({ type: "results-received", payload: { results: [] } });
  dispatch({ type: "refresh-started" });
  dispatch({
    type: "index-progress",
    payload: {
      status: "running",
      phase: "start",
      current: 0,
      total: 0,
      embeddedCount: 0,
      reusedCount: 0,
      message: "Scanning the selected folder for index updates."
    }
  });
  const payload = await desktopSelectFolder(folderPath);
  dispatch({ type: "folder-selected", payload });
  dispatch({ type: "refresh-finished", payload: { label: "Indexed" } });
}

async function chooseFolder() {
  try {
    const folderPath = await pickFolder();
    if (!folderPath) {
      return;
    }
    await selectFolderByPath(folderPath);
  } catch (error) {
    dispatch({ type: "refresh-finished", payload: { label: "Index failed" } });
    dispatch({ type: "task-message", payload: error.message });
  }
}

async function loadAutomationFolderPath() {
  const folderPath = elements.automationFolderInput.value.trim();
  if (!folderPath) {
    return;
  }
  try {
    await selectFolderByPath(folderPath);
  } catch (error) {
    dispatch({ type: "refresh-finished", payload: { label: "Index failed" } });
    dispatch({ type: "task-message", payload: error.message });
  }
}

async function loadAutomationImagePath() {
  const filePath = elements.automationImageInput.value.trim();
  if (!filePath) {
    return;
  }
  try {
    const response = await fetch(toAssetUrl(filePath));
    if (!response.ok) {
      throw new Error("The selected image is not available.");
    }
    const blob = await response.blob();
    const fileName = filePath.split(/[\\/]/).at(-1) || "image-query";
    const file = new File([blob], fileName, { type: blob.type || "" });
    await runUploadedImageSearch(file);
  } catch (error) {
    dispatch({ type: "task-message", payload: error.message });
  }
}

async function runStage2() {
  const guardMessage = resolveStage2LaunchGuard({
    setupStatus: state.setup.status,
    activeFolder: state.activeFolder,
    indexingStatus: state.indexing.status,
    indexedImageCount: indexedImageCount(),
    stage2Status: state.stage2.status
  });
  if (guardMessage) {
    dispatch({
      type: "stage2-progress",
      payload: {
        status: "failed",
        phase: "validate",
        message: guardMessage
      }
    });
    return;
  }
  try {
    dispatch({
      type: "stage2-progress",
      payload: {
        status: "running",
        phase: "validate",
        current: 0,
        total: 0,
        message: "Checking the active folder for Stage 2 adaptation."
      }
    });
    const payload = await desktopRunStage2();
    dispatch({ type: "folder-selected", payload });
    dispatch({ type: "task-message", payload: payload.lastTaskMessage ?? "Stage 2 adaptation is ready." });
    if (currentQuery.kind !== "none") {
      await rerunCurrentSearch();
    }
  } catch (error) {
    dispatch({
      type: "stage2-progress",
      payload: {
        status: "failed",
        phase: "failed",
        message: error.message
      }
    });
    dispatch({ type: "task-message", payload: error.message });
  }
}

async function runTextSearch(queryText) {
  const text = queryText.trim();
  if (!text) {
    clearCurrentQuery();
    dispatch({ type: "results-received", payload: { results: [] } });
    return;
  }

  const payload = normalizeResultsPayload(await desktopSearchText(text, normalizeLimit()));
  replaceCurrentQuery({ kind: "text", text });
  dispatch({ type: "results-received", payload });
}

async function runUploadedImageSearch(file, { previewUrl = null, previewUrlOwned = false } = {}) {
  if (!file) {
    return;
  }

  const resolvedPreviewUrl = previewUrl ?? URL.createObjectURL(file);
  const ownsPreviewUrl = previewUrl ? previewUrlOwned : true;
  try {
    const imageBase64 = await blobToBase64(file);
    const payload = normalizeResultsPayload(
      await desktopSearchUploadedImage(imageBase64, file.name || "pasted-image", normalizeLimit())
    );
    replaceCurrentQuery({
      kind: "upload",
      fileName: file.name || "Pasted image",
      file,
      previewUrl: resolvedPreviewUrl,
      previewUrlOwned: ownsPreviewUrl
    });
    dispatch({ type: "results-received", payload });
    dispatch({ type: "task-message", payload: `Showing matches for ${file.name || "the selected image"}.` });
  } catch (error) {
    if (!previewUrl && resolvedPreviewUrl.startsWith("blob:")) {
      URL.revokeObjectURL(resolvedPreviewUrl);
    }
    throw error;
  }
}

async function runSimilarSearch(item) {
  const payload = normalizeResultsPayload(await desktopSearchSimilar(item.relativePath ?? "", normalizeLimit()));
  replaceCurrentQuery({
    kind: "similar",
    relativePath: item.relativePath ?? "",
    fileName: item.fileName ?? item.name ?? "this image",
    previewUrl: item.thumbnailUrl ?? item.fullUrl ?? "",
    previewUrlOwned: false
  });
  dispatch({ type: "results-received", payload });
  dispatch({ type: "task-message", payload: `Showing images similar to ${currentQuery.fileName}.` });
}

async function rerunCurrentSearch() {
  if (currentQuery.kind === "similar" && currentQuery.relativePath) {
    const payload = normalizeResultsPayload(
      await desktopSearchSimilar(currentQuery.relativePath, normalizeLimit())
    );
    dispatch({ type: "results-received", payload });
    return;
  }
  if (currentQuery.kind === "upload" && currentQuery.file) {
    await runUploadedImageSearch(currentQuery.file, {
      previewUrl: currentQuery.previewUrl,
      previewUrlOwned: currentQuery.previewUrlOwned
    });
    return;
  }
  if (currentQuery.kind === "text" && currentQuery.text) {
    const payload = normalizeResultsPayload(await desktopSearchText(currentQuery.text, normalizeLimit()));
    dispatch({ type: "results-received", payload });
    return;
  }
  if (elements.searchInput.value.trim()) {
    await runTextSearch(elements.searchInput.value);
  }
}

async function refreshSearchAfterDelete(deletedRelativePaths) {
  const deletedSet = new Set(deletedRelativePaths);
  selectedPaths = new Set([...selectedPaths].filter((path) => !deletedSet.has(path)));
  if (currentQuery.kind === "similar" && deletedSet.has(currentQuery.relativePath)) {
    clearCurrentQuery();
    dispatch({ type: "results-received", payload: { results: [] } });
    return;
  }
  await rerunCurrentSearch();
}

async function handlePasteImage(event) {
  if (!state.activeFolder) {
    return;
  }
  const file = extractImageFileFromClipboardItems(event.clipboardData?.items);
  if (!file) {
    return;
  }
  event.preventDefault();
  event.stopPropagation();
  try {
    await runUploadedImageSearch(file);
  } catch (error) {
    dispatch({ type: "task-message", payload: error.message });
  }
}

async function deleteActiveImage() {
  if (!lightboxItem?.relativePath || elements.lightboxDelete.disabled) {
    return;
  }
  const confirmed = await requestTrashConfirmation({});
  if (!confirmed) {
    return;
  }

  elements.lightboxDelete.disabled = true;
  try {
    const payload = await desktopDeleteImage(lightboxItem.relativePath);
    closeLightbox();
    await syncRuntimeStatusSnapshot();
    await refreshSearchAfterDelete([payload.relativePath]);
    dispatch({ type: "task-message", payload: payload.message ?? "Image moved to the Trash." });
  } catch (error) {
    elements.lightboxDelete.disabled = false;
    dispatch({ type: "task-message", payload: error.message });
  }
}

async function deleteSelectedImages() {
  selectedPaths = pruneSelectionToVisibleResults(state.results, selectedPaths);
  if (!selectedPaths.size || elements.selectionDelete.disabled) {
    return;
  }
  const confirmed = await requestTrashConfirmation({ count: selectedPaths.size });
  if (!confirmed) {
    return;
  }

  elements.selectionDelete.disabled = true;
  try {
    const payload = await desktopDeleteImages([...selectedPaths]);
    await syncRuntimeStatusSnapshot();
    await refreshSearchAfterDelete((payload.deleted ?? []).map((item) => item.relativePath));
    resetSelection({ keepMode: true });
    dispatch({ type: "task-message", payload: payload.message ?? "Selected images moved to the Trash." });
  } catch (error) {
    dispatch({ type: "task-message", payload: error.message });
  } finally {
    updateSelectionBar(state.results);
    renderSelectionState();
  }
}

async function registerSetupListeners() {
  await Promise.all([
    listen(SETUP_PROGRESS_EVENT, (event) => {
      dispatch({ type: "setup-progress", payload: event.payload });
    }),
    listen(SETUP_LOG_EVENT, (event) => {
      dispatch({ type: "setup-log", payload: event.payload });
    }),
    listen(SETUP_ERROR_EVENT, (event) => {
      setBootstrapPolling(false);
      dispatch({ type: "setup-failed", payload: event.payload?.message ?? "Desktop runtime startup failed." });
    }),
    listen(RUNTIME_READY_EVENT, async () => {
      setBootstrapPolling(false);
      await syncBootstrapState();
      await handleRuntimeReady();
      render();
    })
  ]);
}

async function registerDesktopListeners() {
  await listen(INDEX_PROGRESS_EVENT, (event) => {
    dispatch({ type: "index-progress", payload: event.payload });
  });
}

function startAutoRefreshLoop() {
  if (autoRefreshHandle) {
    return;
  }
  autoRefreshHandle = window.setInterval(() => {
    if (
      !state.activeFolder ||
      state.setup.status !== "ready" ||
      state.refresh.isRunning ||
      state.indexing.status === "running" ||
      state.stage2.status === "running"
    ) {
      return;
    }

    refreshFolder({ lightweight: true, silent: true }).catch(() => {
      // The failure state is already reduced into UI state.
    });
  }, AUTO_REFRESH_INTERVAL_MS);
}

async function initialize() {
  await registerSetupListeners();
  await registerDesktopListeners();
  startAutoRefreshLoop();
  await loadAutomationToolsStatus();
  await syncBootstrapState();
  if (state.setup.onboardingRequired) {
    if (state.setup.status === "running" || state.setup.status === "cancelling") {
      onboardingStep = "install";
      render();
      setBootstrapPolling(true);
      return;
    }

    if (state.setup.status === "ready") {
      await handleRuntimeReady();
      onboardingStep = "complete";
      render();
    }
    return;
  }

  if (state.setup.status === "ready") {
    await handleRuntimeReady();
    return;
  }
  if (state.setup.status === "running" || state.setup.status === "cancelling") {
    setBootstrapPolling(true);
    return;
  }
  await triggerRuntimeStart();
}

elements.setupLogList.addEventListener("scroll", () => {
  setupLogsPinnedToBottom = nextSetupLogPinState({
    scrollTop: elements.setupLogList.scrollTop,
    clientHeight: elements.setupLogList.clientHeight,
    scrollHeight: elements.setupLogList.scrollHeight
  });
});
elements.refreshButton.addEventListener("click", refreshFolder);
elements.settingsButton.addEventListener("click", async () => {
  await loadAutomationToolsStatus();
  dispatch({ type: "settings-opened" });
});
elements.settingsBackButton.addEventListener("click", () => dispatch({ type: "settings-closed" }));
elements.chooseFolderButton.addEventListener("click", chooseFolder);
elements.automationFolderSubmit.addEventListener("click", loadAutomationFolderPath);
elements.automationFolderInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    loadAutomationFolderPath();
  }
});
elements.runStage2Button.addEventListener("click", runStage2);
elements.openUninstallerButton.addEventListener("click", async () => {
  try {
    await openUninstaller();
  } catch (error) {
    dispatch({ type: "task-message", payload: error.message });
  }
});
elements.searchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    if (elements.searchInput.value.trim()) {
      await runTextSearch(elements.searchInput.value);
      return;
    }
    if (currentQuery.kind !== "none") {
      await rerunCurrentSearch();
      return;
    }
    await runTextSearch("");
  } catch (error) {
    dispatch({ type: "task-message", payload: error.message });
  }
});
elements.clearQueryChipButton.addEventListener("click", () => {
  clearCurrentQuery({ clearResults: true });
  render();
});
elements.automationImageSubmit.addEventListener("click", loadAutomationImagePath);
elements.automationImageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    loadAutomationImagePath();
  }
});
elements.searchInput.addEventListener("paste", handlePasteImage);
elements.searchForm.addEventListener("paste", handlePasteImage);
elements.resultLimit.addEventListener("change", async () => {
  if (!state.activeFolder || currentQuery.kind === "none") {
    return;
  }
  try {
    await rerunCurrentSearch();
  } catch (error) {
    dispatch({ type: "task-message", payload: error.message });
  }
});
elements.selectionToggle.addEventListener("click", () => {
  selectionMode = !selectionMode;
  if (!selectionMode) {
    resetSelection();
  }
  updateSelectionBar(state.results);
  renderSelectionState();
});
elements.selectionSelectAll.addEventListener("click", () => {
  selectedPaths = toggleAllVisibleResults(state.results, selectedPaths);
  updateSelectionBar(state.results);
  renderSelectionState();
});
elements.selectionClear.addEventListener("click", () => {
  resetSelection({ keepMode: true });
  updateSelectionBar(state.results);
  renderSelectionState();
});
elements.selectionDelete.addEventListener("click", async () => {
  await deleteSelectedImages();
});
elements.results.addEventListener("click", (event) => {
  const card = event.target.closest(".result-card");
  if (!card) {
    return;
  }

  const item = state.results.find((result) => result.relativePath === (card.dataset.relativePath || ""));
  if (!item) {
    return;
  }

  if (selectionMode) {
    selectedPaths = toggleResultSelection(selectedPaths, item.relativePath);
    updateSelectionBar(state.results);
    renderSelectionState();
    return;
  }

  openLightbox(item).catch((error) => {
    dispatch({ type: "task-message", payload: error.message });
  });
});
elements.lightboxClose.addEventListener("click", closeLightbox);
elements.lightboxDelete.addEventListener("click", async () => {
  await deleteActiveImage();
});
elements.lightboxSimilar.addEventListener("click", async () => {
  if (!lightboxItem?.relativePath || elements.lightboxSimilar.disabled) {
    return;
  }
  try {
    await runSimilarSearch(lightboxItem);
    closeLightbox();
  } catch (error) {
    dispatch({ type: "task-message", payload: error.message });
  }
});
elements.lightboxInfoToggle.addEventListener("click", () => {
  const nextHidden = !elements.lightboxMeta.hidden;
  elements.lightboxMeta.hidden = nextHidden;
  elements.lightboxInfoToggle.setAttribute("aria-expanded", String(!nextHidden));
});
elements.lightbox.addEventListener("click", (event) => {
  if (event.target === elements.lightbox) {
    closeLightbox();
  }
});
elements.confirmDialog.addEventListener("click", (event) => {
  if (event.target === elements.confirmDialog) {
    settleTrashConfirmation(false);
  }
});
elements.confirmDialogCancel.addEventListener("click", () => {
  settleTrashConfirmation(false);
});
elements.confirmDialogAccept.addEventListener("click", () => {
  settleTrashConfirmation(true);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !elements.confirmDialog.hidden) {
    settleTrashConfirmation(false);
    return;
  }
  if (event.key === "Escape" && !elements.lightbox.hidden) {
    closeLightbox();
  }
});
elements.startSetupButton.addEventListener("click", () => {
  onboardingStep = "install";
  dispatch({
    type: "bootstrap-snapshot",
    payload: {
      status: "running",
      currentStep: 0,
      totalSteps: state.setup.totalSteps,
      message: "Preparing the local desktop runtime.",
      steps: state.setup.steps.map((step) => ({ ...step, status: "idle" })),
      logs: ["Starting SemanticGallery desktop runtime."]
    }
  });
  triggerRuntimeStart({ force: true }).catch(() => {
    // The error is already reduced into state.
  });
});
elements.cancelSetupButton.addEventListener("click", async () => {
  try {
    const payload = await cancelRuntimeSetup();
    dispatch({ type: "bootstrap-snapshot", payload });
    setBootstrapPolling(payload.status === "running" || payload.status === "cancelling");
    if (payload.status === "idle") {
      onboardingStep = "intro";
      render();
    }
  } catch (error) {
    dispatch({ type: "setup-failed", payload: error.message });
    setBootstrapPolling(false);
  }
});
elements.finishSetupButton.addEventListener("click", async () => {
  try {
    const payload = await completeOnboarding();
    dispatch({ type: "bootstrap-snapshot", payload });
    onboardingStep = null;
    render();
  } catch (error) {
    dispatch({ type: "setup-failed", payload: error.message });
  }
});

render();
initialize();
