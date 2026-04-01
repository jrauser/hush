/**
 * Hush Chrome Extension — background service worker.
 *
 * Manages: WebSocket connection to the desktop app, offscreen document
 * lifecycle for tab capture, and tab audio mute/unmute.
 *
 * Frame capture itself happens in the offscreen document (offscreen.js),
 * because MV3 service workers lack DOM and media APIs.
 */

// ── Configuration ────────────────────────────────────────────────────────────

const DEFAULT_SERVER = "localhost:9099";
const HEARTBEAT_INTERVAL_MS = 5000;
const MAX_BACKOFF_MS = 30000;

// ── State ────────────────────────────────────────────────────────────────────

let ws = null;
let state = {
  enabled: false,      // user has clicked "Enable on this tab"
  connected: false,    // WebSocket to app is open
  mode: "idle",        // "idle" or "active"
  intervalSec: 2,
  muted: false,
  tabId: null,
  tabUrl: "",
  streamReady: false,  // offscreen document has a live MediaStream
  frameCount: 0,
};

let heartbeatTimer = null;
let backoffMs = 1000;

// ── WebSocket connection ─────────────────────────────────────────────────────

function connect() {
  if (!state.enabled) return;
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  const url = `ws://${DEFAULT_SERVER}/ws`;
  console.log("[hush] connecting to", url);
  ws = new WebSocket(url);

  ws.onopen = () => {
    console.log("[hush] WebSocket connected");
    state.connected = true;
    backoffMs = 1000;
    updateIcon();
    startHeartbeat();
    sendStatus();
  };

  ws.onmessage = (event) => {
    handleCommand(event.data);
  };

  ws.onclose = () => {
    console.log("[hush] WebSocket disconnected");
    state.connected = false;
    state.mode = "idle";
    updateIcon();
    stopHeartbeat();
    tellOffscreen("stopCapture");
    revertMute();
    scheduleReconnect();
  };

  ws.onerror = () => {
    // Chrome logs ERR_CONNECTION_REFUSED to the console automatically —
    // this is expected when the desktop app isn't running.  Nothing to do
    // here; onclose fires next and handles reconnect with backoff.
  };
}

function disconnect() {
  state.enabled = false;
  if (ws) {
    ws.onclose = null;
    ws.close();
    ws = null;
  }
  state.connected = false;
  state.mode = "idle";
  stopHeartbeat();
  revertMute();
  updateIcon();
}

function scheduleReconnect() {
  if (!state.enabled) return;
  // Only log at longer intervals to avoid console spam during backoff
  if (backoffMs >= MAX_BACKOFF_MS) {
    console.log("[hush] app not reachable, retrying...");
  }
  setTimeout(connect, backoffMs);
  backoffMs = Math.min(backoffMs * 2, MAX_BACKOFF_MS);
}

function sendJson(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function sendStatus() {
  sendJson({
    event: "status",
    mode: state.mode,
    muted: state.muted,
    interval_sec: state.intervalSec,
  });
}

// ── Heartbeat ────────────────────────────────────────────────────────────────

function startHeartbeat() {
  stopHeartbeat();
  heartbeatTimer = setInterval(sendStatus, HEARTBEAT_INTERVAL_MS);
}

function stopHeartbeat() {
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
    heartbeatTimer = null;
  }
}

// ── Command handling (from app via WebSocket) ────────────────────────────────

function handleCommand(data) {
  let msg;
  try {
    msg = JSON.parse(data);
  } catch (e) {
    console.warn("[hush] malformed command:", data);
    return;
  }

  console.log("[hush] received command:", msg);

  switch (msg.cmd) {
    case "set_mode":
      if (msg.mode === "active") {
        state.intervalSec = msg.interval_sec || state.intervalSec;
        state.mode = "active";
        if (state.streamReady) {
          tellOffscreen("startCapture", { intervalSec: state.intervalSec });
        }
      } else if (msg.mode === "idle") {
        state.mode = "idle";
        tellOffscreen("stopCapture");
      }
      sendStatus();
      break;

    case "mute":
      setTabMuted(true);
      break;

    case "unmute":
      setTabMuted(false);
      break;

    default:
      console.warn("[hush] unknown command:", msg.cmd);
  }
}

// ── Tab audio control ────────────────────────────────────────────────────────

async function setTabMuted(muted) {
  if (state.tabId == null) return;
  try {
    await chrome.tabs.update(state.tabId, { muted });
    state.muted = muted;
    sendStatus();
  } catch (e) {
    console.warn("[hush] failed to set tab muted:", e);
  }
}

function revertMute() {
  if (state.muted && state.tabId != null) {
    chrome.tabs.update(state.tabId, { muted: false }).catch(() => {});
    state.muted = false;
  }
}

// ── User mute override detection ─────────────────────────────────────────────

chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (tabId !== state.tabId || !state.enabled) return;
  if (!changeInfo.mutedInfo) return;

  const { muted, reason } = changeInfo.mutedInfo;
  if (reason === "user") {
    console.log("[hush] user mute override detected, muted:", muted);
    state.muted = muted;
    sendJson({ event: "override", muted });
  }
});

// ── Offscreen document management ────────────────────────────────────────────

function tellOffscreen(type, extra = {}) {
  chrome.runtime.sendMessage({ target: "offscreen", type, ...extra }).catch(() => {
    // Offscreen document may not exist yet
  });
}

async function createOffscreenAndStartStream(streamId) {
  try {
    const hasDoc = await chrome.offscreen.hasDocument();
    if (!hasDoc) {
      await chrome.offscreen.createDocument({
        url: "offscreen.html",
        reasons: ["USER_MEDIA"],
        justification: "Captures tab video frames for ad detection",
      });
    }
    console.log("[hush] offscreen document created, sending stream ID");
    // Small delay to let the offscreen document's listener register
    setTimeout(() => {
      chrome.runtime.sendMessage({
        target: "offscreen",
        type: "startStream",
        streamId: streamId,
        serverUrl: `http://${DEFAULT_SERVER}`,
        tabUrl: state.tabUrl,
      });
    }, 100);
  } catch (e) {
    console.error("[hush] failed to create offscreen document:", e);
  }
}

async function destroyOffscreen() {
  tellOffscreen("shutdown");
  state.streamReady = false;
  try {
    const hasDoc = await chrome.offscreen.hasDocument();
    if (hasDoc) {
      await chrome.offscreen.closeDocument();
    }
  } catch (e) {
    // Ignore — document may already be gone
  }
}

// ── Icon management ──────────────────────────────────────────────────────────

function updateIcon() {
  const color = state.connected ? "green" : "grey";
  chrome.action.setIcon({
    path: {
      16: `icons/icon-${color}-16.png`,
      48: `icons/icon-${color}-48.png`,
    },
  });
}

// ── Tab close listener ───────────────────────────────────────────────────────

chrome.tabs.onRemoved.addListener((tabId) => {
  if (tabId === state.tabId) {
    console.log("[hush] monitored tab closed");
    handleDisable();
  }
});

// ── Enable / disable ─────────────────────────────────────────────────────────

async function handleEnable(msg) {
  state.enabled = true;
  state.tabId = msg.tabId;
  state.tabUrl = msg.tabUrl || "";
  state.frameCount = 0;
  state.streamReady = false;
  backoffMs = 1000;

  await createOffscreenAndStartStream(msg.streamId);
  connect();
}

async function handleDisable() {
  disconnect();
  await destroyOffscreen();
  state.tabId = null;
  state.tabUrl = "";
  state.frameCount = 0;
  state.streamReady = false;
}

// ── Message handler (popup + offscreen) ──────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  // ── Messages from popup ──
  if (msg.type === "getState") {
    sendResponse({ ...state });
    return true;
  }
  if (msg.type === "prepareEnable") {
    // Tear down any stale offscreen document (e.g. from a previous session
    // whose service worker was killed by Chrome) so that the tab's capture
    // slot is free before the popup calls getMediaStreamId().
    destroyOffscreen().then(() => sendResponse({ ok: true }));
    return true;
  }
  if (msg.type === "enable") {
    handleEnable(msg).then(() => sendResponse({ ok: true }));
    return true;
  }
  if (msg.type === "disable") {
    handleDisable().then(() => sendResponse({ ok: true }));
    return true;
  }
  if (msg.type === "overrideMute") {
    setTabMuted(false).then(() => {
      sendJson({ event: "override", muted: false });
      sendResponse({ ok: true });
    });
    return true;
  }

  // ── Messages from offscreen document ──
  if (msg.target === "background") {
    if (msg.type === "captureStatus") {
      if (msg.status === "ready") {
        console.log("[hush] offscreen stream ready");
        state.streamReady = true;
        // If we're already supposed to be active, start capture now
        if (state.connected && state.mode === "active") {
          tellOffscreen("startCapture", { intervalSec: state.intervalSec });
        }
      } else if (msg.status === "error") {
        console.error("[hush] offscreen stream error:", msg.error);
        state.streamReady = false;
      }
    }
    if (msg.type === "framePosted") {
      if (msg.ok) state.frameCount++;
    }
  }
});

// ── Startup ──────────────────────────────────────────────────────────────────

// No auto-connect. The user enables the extension via the popup, which
// obtains the tabCapture stream ID and initiates the connection.
