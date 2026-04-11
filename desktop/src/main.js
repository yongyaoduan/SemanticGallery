import "./styles.css";

import { listen } from "@tauri-apps/api/event";

import { fetchJson, pickFolder } from "./api-client";
import { createInitialState, reduceAction } from "./state";

const SETUP_PROGRESS_EVENT = "semanticgallery://setup-progress";
const SETUP_LOG_EVENT = "semanticgallery://setup-log";
const SETUP_ERROR_EVENT = "semanticgallery://setup-error";

let state = createInitialState();

const elements = {
  setupCard: document.getElementById("setup-card"),
  setupTitle: document.getElementById("setup-title"),
  setupMessage: document.getElementById("setup-message"),
  setupProgressLabel: document.getElementById("setup-progress-label"),
  setupProgressBar: document.getElementById("setup-progress-bar"),
  setupStepList: document.getElementById("setup-step-list"),
  setupLogCount: document.getElementById("setup-log-count"),
  setupLogList: document.getElementById("setup-log-list"),
  activeFolderLabel: document.getElementById("active-folder-label"),
  refreshStatusLabel: document.getElementById("refresh-status-label"),
  lastTaskMessage: document.getElementById("last-task-message"),
  results: document.getElementById("results"),
  resultCardTemplate: document.getElementById("result-card-template"),
  settingsPanel: document.getElementById("settings-panel"),
  refreshButton: document.getElementById("refresh-button"),
  settingsButton: document.getElementById("settings-button"),
  settingsClose: document.getElementById("settings-close"),
  chooseFolderButton: document.getElementById("choose-folder-button"),
  runStage2Button: document.getElementById("run-stage2-button"),
  searchForm: document.getElementById("search-form"),
  searchInput: document.getElementById("search-input")
};

function dispatch(action) {
  state = reduceAction(state, action);
  render();
}

function renderResults(results) {
  elements.results.replaceChildren();
  if (!results.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = state.activeFolder
      ? "No results yet. Try a search after the index is ready."
      : "Choose a folder to start indexing and searching.";
    elements.results.append(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const item of results) {
    const node = elements.resultCardTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".result-name").textContent = item.name ?? item.fileName ?? "Image";
    node.querySelector(".result-path").textContent = item.relativePath ?? item.path ?? "";
    fragment.append(node);
  }
  elements.results.append(fragment);
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
    fill.style.width = step.status === "done" ? "100%" : step.status === "running" ? "62%" : "0%";
    bar.append(fill);

    row.append(meta, bar);
    fragment.append(row);
  }

  elements.setupStepList.append(fragment);
}

function renderSetupLogs() {
  elements.setupLogList.replaceChildren();
  elements.setupLogCount.textContent = `${state.setup.logs.length} lines`;

  if (!state.setup.logs.length) {
    const empty = document.createElement("p");
    empty.className = "setup-log-empty";
    empty.textContent = "No setup log lines yet.";
    elements.setupLogList.append(empty);
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
}

function render() {
  const setupPercent =
    state.setup.totalSteps > 0 ? Math.round((state.setup.currentStep / state.setup.totalSteps) * 100) : 0;

  elements.setupCard.hidden = state.setup.status === "ready" || state.setup.status === "idle";
  elements.setupTitle.textContent =
    state.setup.status === "failed" ? "Desktop runtime needs attention" : "Preparing the desktop runtime";
  elements.setupMessage.textContent = state.setup.message;
  elements.setupProgressLabel.textContent = `${state.setup.currentStep} / ${state.setup.totalSteps}`;
  elements.setupProgressBar.style.width = `${setupPercent}%`;

  elements.activeFolderLabel.textContent = state.activeFolder ?? "Choose a folder from Settings";
  elements.refreshStatusLabel.textContent = state.refresh.label;
  elements.lastTaskMessage.textContent = state.lastTaskMessage;
  elements.settingsPanel.hidden = !state.settingsOpen;
  renderSetupSteps();
  renderSetupLogs();
  renderResults(state.results);
}

async function loadRuntimeStatus() {
  try {
    const payload = await fetchJson("/api/runtime/status");
    dispatch({ type: "runtime-status", payload });
  } catch (error) {
    dispatch({ type: "setup-failed", payload: error.message });
    throw error;
  }
}

async function refreshFolder() {
  dispatch({ type: "refresh-started" });
  try {
    const payload = await fetchJson("/api/folders/refresh", { method: "POST" });
    dispatch({ type: "folder-selected", payload });
    dispatch({ type: "refresh-finished", payload: { label: payload.skipped ? "Up to date" : "Refreshed" } });
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
    dispatch({ type: "refresh-started" });
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
  } catch (error) {
    dispatch({ type: "task-message", payload: error.message });
  }
}

async function searchText(event) {
  event.preventDefault();
  const query = elements.searchInput.value.trim();
  try {
    const payload = await fetchJson(`/api/search?q=${encodeURIComponent(query)}&limit=25`);
    dispatch({ type: "results-received", payload });
  } catch (error) {
    dispatch({ type: "task-message", payload: error.message });
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
      dispatch({ type: "setup-failed", payload: event.payload?.message ?? "Desktop runtime startup failed." });
    })
  ]);
}

async function initialize() {
  await registerSetupListeners();
  dispatch({
    type: "setup-progress",
    payload: {
      task: "check-runtime",
      phase: "start",
      current: 0,
      total: 5,
      message: "Starting the local desktop runtime."
    }
  });

  try {
    await loadRuntimeStatus();
    dispatch({ type: "setup-finished" });
  } catch {
    // The setup error has already been reduced into state.
  }
}

elements.refreshButton.addEventListener("click", refreshFolder);
elements.settingsButton.addEventListener("click", () => dispatch({ type: "settings-opened" }));
elements.settingsClose.addEventListener("click", () => dispatch({ type: "settings-closed" }));
elements.chooseFolderButton.addEventListener("click", chooseFolder);
elements.runStage2Button.addEventListener("click", runStage2);
elements.searchForm.addEventListener("submit", searchText);

render();
initialize();
