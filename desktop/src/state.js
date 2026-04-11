export function createInitialState() {
  return {
    setup: {
      currentStep: 0,
      totalSteps: 5,
      status: "idle",
      message: "Checking local resources and preparing the model runtime."
    },
    refresh: {
      isRunning: false,
      label: "Idle"
    },
    settingsOpen: false,
    activeFolder: null,
    activeEncoderSignature: "stage1",
    lastTaskMessage: "No background task has started yet.",
    results: []
  };
}

export function reduceAction(state, action) {
  if (action.type === "setup-progress") {
    return {
      ...state,
      setup: {
        ...state.setup,
        ...action.payload,
        status: "running"
      }
    };
  }

  if (action.type === "setup-finished") {
    return {
      ...state,
      setup: {
        ...state.setup,
        status: "ready",
        currentStep: state.setup.totalSteps
      }
    };
  }

  if (action.type === "refresh-started") {
    return {
      ...state,
      refresh: {
        isRunning: true,
        label: "Refreshing"
      }
    };
  }

  if (action.type === "refresh-finished") {
    return {
      ...state,
      refresh: {
        isRunning: false,
        label: action.payload?.label ?? "Idle"
      }
    };
  }

  if (action.type === "settings-opened") {
    return { ...state, settingsOpen: true };
  }

  if (action.type === "settings-closed") {
    return { ...state, settingsOpen: false };
  }

  if (action.type === "runtime-status") {
    return {
      ...state,
      activeFolder: action.payload.activeFolder,
      activeEncoderSignature: action.payload.activeEncoderSignature,
      lastTaskMessage: action.payload.lastTaskMessage ?? state.lastTaskMessage
    };
  }

  if (action.type === "folder-selected") {
    return {
      ...state,
      activeFolder: action.payload.activeFolder,
      activeEncoderSignature: action.payload.activeEncoderSignature ?? state.activeEncoderSignature,
      lastTaskMessage: action.payload.lastTaskMessage ?? state.lastTaskMessage
    };
  }

  if (action.type === "results-received") {
    return {
      ...state,
      results: action.payload.results ?? []
    };
  }

  if (action.type === "task-message") {
    return {
      ...state,
      lastTaskMessage: action.payload
    };
  }

  return state;
}
