export const SETUP_LOG_BOTTOM_THRESHOLD_PX = 16;

function distanceFromBottom({ scrollTop = 0, clientHeight = 0, scrollHeight = 0 } = {}) {
  return scrollHeight - (scrollTop + clientHeight);
}

export function nextSetupLogPinState(metrics = {}) {
  return distanceFromBottom(metrics) <= SETUP_LOG_BOTTOM_THRESHOLD_PX;
}

export function shouldAutoScrollSetupLogs({ pinnedToBottom = true, ...metrics } = {}) {
  return Boolean(pinnedToBottom) || nextSetupLogPinState(metrics);
}
