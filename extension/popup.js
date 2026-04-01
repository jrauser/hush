document.addEventListener("DOMContentLoaded", () => {
  const connEl = document.getElementById("connection");
  const modeEl = document.getElementById("mode");
  const intervalEl = document.getElementById("interval");
  const audioEl = document.getElementById("audio");
  const enableBtn = document.getElementById("enable-btn");
  const unmuteBtn = document.getElementById("unmute-btn");

  function updateUI(state) {
    if (!state) {
      connEl.textContent = "Error";
      connEl.className = "value disconnected";
      return;
    }

    // Connection
    if (state.connected) {
      connEl.textContent = "Connected";
      connEl.className = "value connected";
    } else if (state.enabled) {
      connEl.textContent = "Connecting\u2026";
      connEl.className = "value disconnected";
    } else {
      connEl.textContent = "Disconnected";
      connEl.className = "value disconnected";
    }

    // Mode
    if (state.mode === "active") {
      modeEl.textContent = state.streamReady ? "Capturing" : "Starting\u2026";
    } else {
      modeEl.textContent = "Idle";
    }

    // Interval
    intervalEl.textContent = state.mode === "active"
      ? `${state.intervalSec}s`
      : "--";

    // Audio
    audioEl.textContent = state.muted ? "Muted" : "Unmuted";
    audioEl.className = "value " + (state.muted ? "muted" : "unmuted");

    // Enable/Disable button
    if (state.enabled) {
      enableBtn.textContent = "Disable";
      enableBtn.classList.add("disable");
    } else {
      enableBtn.textContent = "Enable on this tab";
      enableBtn.classList.remove("disable");
    }

    // Unmute button only when muted
    unmuteBtn.style.display = state.muted ? "block" : "none";
  }

  function refreshState() {
    chrome.runtime.sendMessage({ type: "getState" }, updateUI);
  }

  // Poll for state updates every second
  refreshState();
  setInterval(refreshState, 1000);

  enableBtn.addEventListener("click", async () => {
    const currentState = await new Promise((resolve) => {
      chrome.runtime.sendMessage({ type: "getState" }, resolve);
    });

    if (currentState && currentState.enabled) {
      // Disable
      chrome.runtime.sendMessage({ type: "disable" }, refreshState);
    } else {
      // Enable — tear down any stale offscreen document first.  If Chrome
      // killed the service worker while a previous capture was running, the
      // offscreen document (and its MediaStream) can outlive the SW, leaving
      // the tab's capture slot occupied.  getMediaStreamId() will fail with
      // "only one stream per tab" until that slot is freed.
      await new Promise((resolve) => {
        chrome.runtime.sendMessage({ type: "prepareEnable" }, resolve);
      });

      // Get the active tab and obtain a capture stream ID.
      // getMediaStreamId() must be called here (popup) where the user
      // gesture context from the button click is available.
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab) {
        connEl.textContent = "No tab found";
        connEl.className = "value disconnected";
        return;
      }

      try {
        const streamId = await chrome.tabCapture.getMediaStreamId({
          targetTabId: tab.id,
        });
        chrome.runtime.sendMessage({
          type: "enable",
          tabId: tab.id,
          tabUrl: tab.url || "",
          streamId: streamId,
        }, refreshState);
      } catch (e) {
        console.error("[hush popup] getMediaStreamId failed:", e);
        connEl.textContent = "Capture failed";
        connEl.className = "value disconnected";
      }
    }
  });

  unmuteBtn.addEventListener("click", () => {
    chrome.runtime.sendMessage({ type: "overrideMute" }, () => {
      window.close();
    });
  });
});
