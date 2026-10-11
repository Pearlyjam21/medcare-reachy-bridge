const STATE_TEXT = {
  connecting_robot: "Connecting to the robot…",
  not_configured: "Not connected yet",
  starting: "Starting…",
  running: "Connected",
  stopping: "Stopping…",
  stopped: "Stopped",
  error: "Error",
};

const form = document.getElementById("config");
const formError = document.getElementById("form-error");
const saveButton = form.querySelector('button[type="submit"]');
const restartButton = document.getElementById("restart");

// Save and Restart both restart the robot's connection, so they send one request at a time: two at once
// (a double click) once left two connections driving the robot. Both buttons wait for the answer.
function oneAtATime(send) {
  return async (event) => {
    event.preventDefault();
    if (saveButton.disabled) return;
    saveButton.disabled = true;
    restartButton.disabled = true;
    try {
      await send();
    } finally {
      saveButton.disabled = false;
      restartButton.disabled = false;
    }
  };
}

function show(status) {
  const state = document.getElementById("state");
  state.textContent = STATE_TEXT[status.state] || status.state;
  state.dataset.state = status.state;
  document.getElementById("detail").textContent = status.detail || "";
  const metrics = document.getElementById("metrics");
  metrics.replaceChildren();
  const rows = [
    ["Robot", status.robot_reachable == null ? null : (status.robot_reachable ? "reachable" : "unreachable")],
    ["Camera feed", status.camera_fps == null ? null : `${status.camera_fps} fps`],
    ["Camera landmarks", status.landmark_fps == null ? null : `${status.landmark_fps} fps`],
    ["Snapshots", status.vision_fps == null ? null : `${status.vision_fps} fps`],
    ["Current step", status.slot_state],
    ["Missing voice clips", status.missing_clips],
  ];
  for (const [label, value] of rows) {
    if (value == null) continue;
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    metrics.append(dt, dd);
  }
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    show(await response.json());
  } catch {
    show({ state: "error", detail: "The app is not responding." });
  }
}

async function loadConfig() {
  const response = await fetch("/api/config", { cache: "no-store" });
  const config = await response.json();
  form.app_url.value = config.app_url || "";
  form.language.value = config.language;
  form.capture_fps.value = config.capture_fps;
  form.checkin_ack.checked = config.checkin_ack !== false;
  form.checkin_gestures.checked = config.checkin_gestures !== false;
  document.getElementById("token-hint").textContent = config.device_token_set
    ? "A robot key is saved. Leave empty to keep it, or paste a new one."
    : "Shown once when you pair the robot in the web app.";
}

form.addEventListener("submit", oneAtATime(async () => {
  formError.textContent = "";
  const body = {
    app_url: form.app_url.value.trim(),
    device_token: form.device_token.value.trim(),
    language: form.language.value,
    capture_fps: Number(form.capture_fps.value),
    checkin_ack: form.checkin_ack.checked,
    checkin_gestures: form.checkin_gestures.checked,
  };
  const response = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    formError.textContent = data.detail || "Could not save.";
    return;
  }
  form.device_token.value = "";
  show(data.status);
  await loadConfig();
}));

restartButton.addEventListener("click", oneAtATime(async () => {
  const response = await fetch("/api/restart", { method: "POST" });
  show(await response.json());
}));

loadConfig().catch(() => { formError.textContent = "Could not load settings."; });
refreshStatus();
setInterval(refreshStatus, 3000);
