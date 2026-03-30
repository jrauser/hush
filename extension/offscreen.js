/**
 * Hush offscreen document — captures tab video frames and POSTs them to the
 * desktop app.  Runs in a hidden document created by the service worker,
 * because MV3 service workers lack DOM/media APIs.
 */

const JPEG_QUALITY = 0.8;
const FRAME_POST_TIMEOUT_MS = 3000;

let mediaStream = null;
let captureTimer = null;
let serverUrl = "";
let tabUrl = "";

// ── Message handling ────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.target !== "offscreen") return;

  switch (msg.type) {
    case "startStream":
      initStream(msg.streamId, msg.serverUrl, msg.tabUrl);
      break;
    case "startCapture":
      startCapture(msg.intervalSec);
      break;
    case "stopCapture":
      stopCapture();
      break;
    case "shutdown":
      shutdown();
      break;
  }
});

// ── Stream initialization ───────────────────────────────────────────────────

async function initStream(streamId, url, tab) {
  serverUrl = url;
  tabUrl = tab || "";

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        mandatory: {
          chromeMediaSource: "tab",
          chromeMediaSourceId: streamId,
          maxWidth: 800,
          maxHeight: 450,
        },
      },
    });
    console.log("[hush offscreen] stream ready");
    chrome.runtime.sendMessage({
      target: "background",
      type: "captureStatus",
      status: "ready",
    });
  } catch (e) {
    console.error("[hush offscreen] getUserMedia failed:", e);
    chrome.runtime.sendMessage({
      target: "background",
      type: "captureStatus",
      status: "error",
      error: e.message,
    });
  }
}

// ── Frame capture loop ──────────────────────────────────────────────────────

function startCapture(intervalSec) {
  stopCapture();
  console.log(`[hush offscreen] capturing every ${intervalSec}s`);
  captureAndPost(); // immediate first frame
  captureTimer = setInterval(captureAndPost, intervalSec * 1000);
}

function stopCapture() {
  if (captureTimer) {
    clearInterval(captureTimer);
    captureTimer = null;
  }
}

async function captureAndPost() {
  if (!mediaStream) return;

  try {
    const track = mediaStream.getVideoTracks()[0];
    if (!track || track.readyState !== "live") {
      console.warn("[hush offscreen] video track not live, stopping");
      stopCapture();
      chrome.runtime.sendMessage({
        target: "background",
        type: "captureStatus",
        status: "error",
        error: "video track ended",
      });
      return;
    }

    const imageCapture = new ImageCapture(track);
    const bitmap = await imageCapture.grabFrame();

    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();

    const blob = await canvas.convertToBlob({
      type: "image/jpeg",
      quality: JPEG_QUALITY,
    });

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), FRAME_POST_TIMEOUT_MS);

    const formData = new FormData();
    formData.append("frame", blob, "frame.jpg");
    formData.append("timestamp", new Date().toISOString());
    formData.append("tab_url", tabUrl);

    try {
      const response = await fetch(`${serverUrl}/frame`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      chrome.runtime.sendMessage({
        target: "background",
        type: "framePosted",
        ok: response.ok,
      });
    } catch (fetchErr) {
      clearTimeout(timeoutId);
      if (fetchErr.name === "AbortError") {
        console.warn("[hush offscreen] frame POST timed out");
      }
      chrome.runtime.sendMessage({
        target: "background",
        type: "framePosted",
        ok: false,
      });
    }
  } catch (e) {
    console.warn("[hush offscreen] capture error:", e);
  }
}

// ── Cleanup ─────────────────────────────────────────────────────────────────

function shutdown() {
  stopCapture();
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
}
