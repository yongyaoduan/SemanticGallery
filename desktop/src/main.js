import "./styles.css";

import { fetchJson, pickFolder } from "./api-client";
import { createInitialState, reduceAction } from "./state";

let state = createInitialState();

const elements = {
  setupCard: document.getElementById("setup-card"),
  setupTitle: document.getElementById("setup-title"),
  setupMessage: document.getElementById("setup-message"),
  setupProgressLabel: document.getElementById("setup-progress-label"),
  setupProgressBar: document.getElementById("setup-progress-bar"),
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

function render() {
  const setupPercent =
    state.setup.totalSteps > 0 ? Math.round((state.setup.currentStep / state.setup.totalSteps) * 100) : 0;

  elements.setupCard.hidden = state.setup.status === "ready" || state.setup.status === "idle";
  elements.setupTitle.textContent = state.setup.status === "running" ? "Preparing the desktop runtime" : "Desktop runtime ready";
  elements.setupMessage.textContent = state.setup.message;
  elements.setupProgressLabel.textContent = `${state.setup.currentStep} / ${state.setup.totalSteps}`;
  elements.setupProgressBar.style.width = `${setupPercent}%`;

  elements.activeFolderLabel.textContent = state.activeFolder ?? "Choose a folder from Settings";
  elements.refreshStatusLabel.textContent = state.refresh.label;
  elements.lastTaskMessage.textContent = state.lastTaskMessage;
  elements.settingsPanel.hidden = !state.settingsOpen;
  renderResults(state.results);
}

async function loadRuntimeStatus() {
  try {
    const payload = await fetchJson("/api/runtime/status");
    dispatch({ type: "runtime-status", payload });
  } catch (error) {
    dispatch({ type: "task-message", payload: error.message });
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

elements.refreshButton.addEventListener("click", refreshFolder);
elements.settingsButton.addEventListener("click", () => dispatch({ type: "settings-opened" }));
elements.settingsClose.addEventListener("click", () => dispatch({ type: "settings-closed" }));
elements.chooseFolderButton.addEventListener("click", chooseFolder);
elements.runStage2Button.addEventListener("click", runStage2);
elements.searchForm.addEventListener("submit", searchText);

dispatch({ type: "setup-progress", payload: { currentStep: 1, totalSteps: 5, message: "Connecting to the local sidecar." } });
loadRuntimeStatus().finally(() => {
  dispatch({ type: "setup-finished" });
});
