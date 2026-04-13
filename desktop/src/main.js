import "./styles.css";

import { listen } from "@tauri-apps/api/event";

import {
  cancelRuntimeSetup,
  completeOnboarding,
  fetchJson,
  openUninstaller,
  pickFolder,
  resetSidecarBaseUrl,
  sidecarBaseUrl,
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
import { buildIndexProgressSummary } from "./index-progress";
import { buildQueryPresentation } from "./query-presentation";
import { nextSetupLogPinState, shouldAutoScrollSetupLogs } from "./setup-log-scroll";
import { buildTrashConfirmationContent } from "./trash-confirmation";

const SETUP_PROGRESS_EVENT = "semanticgallery://setup-progress";
const SETUP_LOG_EVENT = "semanticgallery://setup-log";
const SETUP_ERROR_EVENT = "semanticgallery://setup-error";
const RUNTIME_READY_EVENT = "semanticgallery://runtime-ready";
const DEFAULT_LIMIT = 25;

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
  settingsIndexProgressLabel: document.getElementById("settings-index-progress-label"),
  settingsIndexProgressBar: document.getElementById("settings-index-progress-bar"),
  settingsIndexProgressTimeRow: document.getElementById("settings-index-progress-time-row"),
  settingsIndexProgressElapsed: document.getElementById("settings-index-progress-elapsed"),
  settingsIndexProgressRemaining: document.getElementById("settings-index-progress-remaining"),
  settingsIndexMessage: document.getElementById("settings-index-message"),
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
  runStage2Button: document.getElementById("run-stage2-button"),
  searchForm: document.getElementById("search-form"),
  searchInput: document.getElementById("search-input"),
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

function withLimit(url) {
  const nextUrl = new URL(url);
  nextUrl.searchParams.set("limit", String(normalizeLimit()));
  return nextUrl.toString();
}

async function normalizeResultsPayload(payload) {
  const baseUrl = await sidecarBaseUrl();
  const resolveMediaUrl = (path) => (path ? new URL(path, `${baseUrl}/`).toString() : "");
  return {
    ...payload,
    results: (payload.results ?? []).map((item) => ({
      ...item,
      thumbnailUrl: resolveMediaUrl(item.thumbnailUrl),
      fullUrl: resolveMediaUrl(item.fullUrl),
      metadataUrl: resolveMediaUrl(item.metadataUrl),
      deleteUrl: resolveMediaUrl(item.deleteUrl),
      similarUrl: resolveMediaUrl(item.similarUrl)
    }))
  };
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
  const payload = await fetchJson("/api/runtime/status");
  dispatch({ type: "runtime-status", payload });
  return payload;
}

async function handleRuntimeReady() {
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
  runtimeEventSource.addEventListener("folder-index-progress", (event) => {
    dispatch({ type: "index-progress", payload: JSON.parse(event.data) });
  });
  runtimeEventSource.addEventListener("folder-refresh-failed", (event) => {
    const payload = JSON.parse(event.data);
    dispatch({ type: "task-message", payload: payload.message ?? "Folder refresh failed." });
  });
  runtimeEventSource.addEventListener("folder-refreshed", async () => {
    try {
      await syncRuntimeStatusSnapshot();
      if (currentQuery.kind !== "none") {
        await rerunCurrentSearch();
      }
    } catch (error) {
      dispatch({ type: "task-message", payload: error.message });
    }
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
  elements.lightboxImage.src = item.fullUrl;
  elements.lightboxImage.alt = item.name ?? item.fileName ?? "Image";
  elements.lightboxMeta.hidden = true;
  elements.lightboxInfoToggle.setAttribute("aria-expanded", "false");
  elements.lightboxFilename.textContent = item.fileName ?? "";
  elements.lightboxPath.textContent = "";
  elements.lightboxSize.textContent = "Loading";
  elements.lightboxDimensions.textContent = "Loading";
  elements.lightboxTimeLabel.textContent = "Time";
  elements.lightboxTime.textContent = "Loading";
  elements.lightboxDelete.disabled = !item.deleteUrl;
  elements.lightboxSimilar.disabled = !item.similarUrl;
  elements.lightbox.hidden = false;
  updateBodyLock();

  if (activeMetadataController) {
    activeMetadataController.abort();
  }
  activeMetadataController = new AbortController();

  try {
    const metadata = await fetchJson(item.metadataUrl, { signal: activeMetadataController.signal });
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
    node.querySelector(".result-thumb").src = item.thumbnailUrl ?? item.fullUrl ?? "";
    node.querySelector(".result-thumb").alt = item.name ?? item.fileName ?? "Image";
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
  elements.settingsIndexProgressLabel.textContent = indexSummary.countLabel;
  elements.settingsIndexProgressBar.style.width = `${indexSummary.percent}%`;
  const showProgressTiming = Boolean(indexSummary.elapsedLabel || indexSummary.remainingLabel);
  elements.settingsIndexProgressTimeRow.hidden = !showProgressTiming;
  elements.settingsIndexProgressTimeRow.style.display = showProgressTiming ? "flex" : "none";
  elements.settingsIndexProgressElapsed.textContent = indexSummary.elapsedLabel;
  elements.settingsIndexProgressRemaining.textContent = indexSummary.remainingLabel;
  elements.settingsIndexMessage.textContent = state.indexing.message;

  elements.refreshButton.disabled = !state.activeFolder || state.refresh.isRunning;
  elements.searchInput.disabled = !state.activeFolder;
  elements.resultLimit.disabled = !state.activeFolder;
  elements.selectionToggle.disabled = !state.results.length;
  elements.chooseFolderButton.disabled = state.setup.status !== "ready" || state.indexing.status === "running";
  elements.runStage2Button.disabled =
    state.setup.status !== "ready" || !state.activeFolder || state.indexing.status === "running";
  elements.searchInput.placeholder = state.activeFolder
    ? "Describe the photo you want to find"
    : "Choose a folder first";
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
  renderSetupLogs();
  renderResults(state.results);
}

async function loadRuntimeStatus() {
  try {
    const payload = await fetchJson("/api/runtime/status");
    dispatch({ type: "runtime-status", payload });
    dispatch({ type: "setup-finished" });
  } catch (error) {
    dispatch({ type: "setup-failed", payload: error.message });
    throw error;
  }
}

async function refreshFolder() {
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
  try {
    const payload = await fetchJson("/api/folders/refresh", { method: "POST" });
    dispatch({ type: "folder-selected", payload });
    dispatch({ type: "refresh-finished", payload: { label: payload.skipped ? "Up to date" : "Refreshed" } });
    if (currentQuery.kind !== "none") {
      await rerunCurrentSearch();
    }
  } catch (error) {
    dispatch({ type: "refresh-finished", payload: { label: "Refresh failed" } });
    dispatch({ type: "task-message", payload: error.message });
  }
}

async function chooseFolder() {
  try {
    const folderPath = await pickFolder();
    if (!folderPath) {
      return;
    }
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
    const payload = await fetchJson("/api/folders/select", {
      method: "POST",
      body: JSON.stringify({ folderPath })
    });
    dispatch({ type: "folder-selected", payload });
    dispatch({ type: "refresh-finished", payload: { label: "Indexed" } });
  } catch (error) {
    dispatch({ type: "refresh-finished", payload: { label: "Index failed" } });
    dispatch({ type: "task-message", payload: error.message });
  }
}

async function runStage2() {
  try {
    dispatch({ type: "task-message", payload: "Running Stage 2 adaptation..." });
    const payload = await fetchJson("/api/stage2/run", { method: "POST" });
    dispatch({ type: "folder-selected", payload });
    dispatch({ type: "task-message", payload: payload.lastTaskMessage ?? "Stage 2 adaptation is ready." });
    if (currentQuery.kind !== "none") {
      await rerunCurrentSearch();
    }
  } catch (error) {
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

  const payload = await normalizeResultsPayload(
    await fetchJson(`/api/search?q=${encodeURIComponent(text)}&limit=${normalizeLimit()}`)
  );
  replaceCurrentQuery({ kind: "text", text });
  dispatch({ type: "results-received", payload });
}

async function runUploadedImageSearch(file, { previewUrl = null, previewUrlOwned = false } = {}) {
  if (!file) {
    return;
  }

  const resolvedPreviewUrl = previewUrl ?? URL.createObjectURL(file);
  const ownsPreviewUrl = previewUrl ? previewUrlOwned : true;
  const formData = new FormData();
  formData.append("image", file, file.name || "pasted-image");
  try {
    const payload = await normalizeResultsPayload(
      await fetchJson(`/api/search/image?limit=${normalizeLimit()}`, {
        method: "POST",
        body: formData
      })
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
  const queryUrl = withLimit(item.similarUrl);
  const payload = await normalizeResultsPayload(await fetchJson(queryUrl));
  replaceCurrentQuery({
    kind: "similar",
    url: item.similarUrl,
    relativePath: item.relativePath ?? "",
    fileName: item.fileName ?? item.name ?? "this image",
    previewUrl: item.thumbnailUrl ?? item.fullUrl ?? "",
    previewUrlOwned: false
  });
  dispatch({ type: "results-received", payload });
  dispatch({ type: "task-message", payload: `Showing images similar to ${currentQuery.fileName}.` });
}

async function rerunCurrentSearch() {
  if (currentQuery.kind === "similar" && currentQuery.url) {
    const payload = await normalizeResultsPayload(await fetchJson(withLimit(currentQuery.url)));
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
    const payload = await normalizeResultsPayload(
      await fetchJson(`/api/search?q=${encodeURIComponent(currentQuery.text)}&limit=${normalizeLimit()}`)
    );
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
  if (!lightboxItem?.deleteUrl || elements.lightboxDelete.disabled) {
    return;
  }
  const confirmed = await requestTrashConfirmation({});
  if (!confirmed) {
    return;
  }

  elements.lightboxDelete.disabled = true;
  try {
    const payload = await fetchJson(lightboxItem.deleteUrl, { method: "DELETE" });
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
    const payload = await fetchJson("/api/images/batch-delete", {
      method: "POST",
      body: JSON.stringify({ paths: [...selectedPaths] })
    });
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

async function initialize() {
  await registerSetupListeners();
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
elements.settingsButton.addEventListener("click", () => dispatch({ type: "settings-opened" }));
elements.settingsBackButton.addEventListener("click", () => dispatch({ type: "settings-closed" }));
elements.chooseFolderButton.addEventListener("click", chooseFolder);
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
  if (!lightboxItem?.similarUrl || elements.lightboxSimilar.disabled) {
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
