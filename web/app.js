(() => {
  const connDot = document.getElementById("conn-dot");
  const statusBadge = document.getElementById("status-badge");
  const fpsCounter = document.getElementById("fps-counter");
  const persistCheckbox = document.getElementById("persist-checkbox");
  const clearProfileBtn = document.getElementById("clear-profile-btn");
  const urlInput = document.getElementById("url-input");
  const openUrlBtn = document.getElementById("open-url-btn");
  const observeBtn = document.getElementById("observe-btn");
  const takeoverText = document.getElementById("takeover-text");
  const takeoverEnter = document.getElementById("takeover-enter");
  const takeoverTypeBtn = document.getElementById("takeover-type-btn");
  const goalInput = document.getElementById("goal-input");
  const modeSelect = document.getElementById("mode-select");
  const runGoalBtn = document.getElementById("run-goal-btn");
  const pauseBtn = document.getElementById("pause-btn");
  const resumeBtn = document.getElementById("resume-btn");
  const stopBtn = document.getElementById("stop-btn");
  const approvalBox = document.getElementById("approval-box");
  const approvalLabel = document.getElementById("approval-label");
  const approveBtn = document.getElementById("approve-btn");
  const rejectBtn = document.getElementById("reject-btn");
  const eventLog = document.getElementById("event-log");
  const clearLogBtn = document.getElementById("clear-log-btn");
  const liveUrl = document.getElementById("live-url");
  const tabStrip = document.getElementById("tab-strip");
  const viewportImg = document.getElementById("viewport-img");
  const targetBox = document.getElementById("target-box");
  const targetLabel = document.getElementById("target-label");
  const emptyState = document.getElementById("empty-state");

  let ws = null;
  let pendingStepId = null;
  let totalFrames = 0;
  let framesThisSecond = 0;
  let currentFps = 0;
  let lastUrl = "";
  let lastTabsJson = "";

  setInterval(() => {
    currentFps = framesThisSecond;
    framesThisSecond = 0;
    fpsCounter.textContent = `${currentFps} fps · ${totalFrames} frames`;
  }, 1000);

  function addLog(text, kind = "info") {
    const div = document.createElement("div");
    div.className = `log-item ${kind}`;
    const time = new Date().toLocaleTimeString();
    div.textContent = `[${time}] ${text}`;
    eventLog.prepend(div);
  }

  function send(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  function renderTabs(tabs) {
    const json = JSON.stringify(tabs || []);
    if (json === lastTabsJson) return;
    lastTabsJson = json;
    tabStrip.innerHTML = "";
    (tabs || []).forEach((t) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `tab-pill${t.active ? " active" : ""}`;
      btn.textContent = `#${t.index + 1} ${(t.title || t.url || "Tab").slice(0, 22)}`;
      btn.onclick = () => send({ type: "switch_tab", index: t.index });
      tabStrip.appendChild(btn);
    });
  }

  function showTargetBox(rect, label) {
    if (!rect || !viewportImg.clientWidth) {
      targetBox.classList.add("hidden");
      return;
    }
    const imgRect = viewportImg.getBoundingClientRect();
    const wrapRect = viewportImg.parentElement.getBoundingClientRect();
    const scaleX = imgRect.width / 1280;
    const scaleY = imgRect.height / 960;
    targetBox.style.left = `${imgRect.left - wrapRect.left + rect.x * scaleX}px`;
    targetBox.style.top = `${imgRect.top - wrapRect.top + rect.y * scaleY}px`;
    targetBox.style.width = `${Math.max(12, rect.w * scaleX)}px`;
    targetBox.style.height = `${Math.max(12, rect.h * scaleY)}px`;
    targetLabel.textContent = label || "";
    targetBox.classList.remove("hidden");
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.onopen = () => {
      connDot.classList.add("connected");
      addLog("Connected to Velox WebSocket server.");
    };
    ws.onclose = () => {
      connDot.classList.remove("connected");
      setTimeout(connect, 1500);
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "frame") {
        // Direct DOM img.src update — zero framework re-render overhead
        emptyState.classList.add("hidden");
        viewportImg.src = `data:image/jpeg;base64,${msg.image}`;
        totalFrames += 1;
        framesThisSecond += 1;
        if (msg.url && msg.url !== lastUrl) {
          lastUrl = msg.url;
          liveUrl.textContent = `${msg.title ? msg.title + " — " : ""}${msg.url}`;
        }
        if (msg.tabs) renderTabs(msg.tabs);
      } else if (msg.type === "status") {
        statusBadge.textContent = msg.state;
        addLog(`Status -> ${msg.state}${msg.detail ? " (" + msg.detail + ")" : ""}`);
      } else if (msg.type === "step") {
        const label = msg.action?.label || msg.action?.name || "";
        showTargetBox(msg.rect, label);
        if (msg.status === "ok" || msg.status === "error") {
          addLog(
            `Step #${msg.n} [${msg.status}] ${label} -> ${msg.error || msg.result || ""}`,
            msg.status === "error" ? "error" : "ok"
          );
        }
      } else if (msg.type === "approval") {
        pendingStepId = msg.step_id;
        approvalLabel.textContent = `${msg.action_label} (${msg.reason || "risky action"})`;
        approvalBox.classList.remove("hidden");
        addLog(`Approval required: ${msg.action_label}`, "error");
      } else if (msg.type === "observation") {
        const lines = msg.elements.map((e) => `[${e.id}] <${e.tag} role=${e.role}> ${e.label}`).join("\n");
        addLog(`DOM Observation (${msg.element_count} interactive elements on ${msg.url}):\n${lines}`, "ok");
      } else if (msg.type === "result") {
        const summary = msg.summary || msg.result?.summary || JSON.stringify(msg, null, 2);
        addLog(`FINAL RESULT:\n${summary}`, "ok");
      } else if (msg.type === "notice") {
        addLog(msg.message, msg.level === "error" ? "error" : "info");
      }
    };
  }

  openUrlBtn.onclick = () => {
    send({ type: "open_url", url: urlInput.value, persist: persistCheckbox.checked });
  };

  observeBtn.onclick = () => {
    send({ type: "observe" });
  };

  runGoalBtn.onclick = () => {
    send({
      type: "run_goal",
      goal: goalInput.value,
      start_url: urlInput.value,
      mode: modeSelect.value,
      persist: persistCheckbox.checked,
    });
  };

  pauseBtn.onclick = () => send({ type: "pause" });
  resumeBtn.onclick = () => send({ type: "resume" });
  stopBtn.onclick = () => send({ type: "stop" });

  approveBtn.onclick = () => {
    if (pendingStepId) send({ type: "approve", step_id: pendingStepId });
    approvalBox.classList.add("hidden");
  };
  rejectBtn.onclick = () => {
    if (pendingStepId) send({ type: "reject", step_id: pendingStepId });
    approvalBox.classList.add("hidden");
  };

  takeoverTypeBtn.onclick = () => {
    if (!takeoverText.value) return;
    send({ type: "type", text: takeoverText.value, press_enter: takeoverEnter.checked });
    takeoverText.value = "";
  };

  clearProfileBtn.onclick = () => send({ type: "clear_profile" });
  clearLogBtn.onclick = () => { eventLog.innerHTML = ""; };

  // Direct click on live browser viewport -> maps to 1280x960 coordinates & focuses viewport for direct typing
  viewportImg.addEventListener("click", (e) => {
    viewportImg.focus();
    const r = viewportImg.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const x = ((e.clientX - r.left) / r.width) * 1280;
    const y = ((e.clientY - r.top) / r.height) * 960;
    send({ type: "click", x, y });
  });

  // Direct keyboard typing when the live browser viewport is focused
  viewportImg.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "v") {
      // Let the paste event handle clipboard text
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (
      e.key.length === 1 ||
      ["Backspace", "Enter", "Tab", "Escape", "Delete", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)
    ) {
      e.preventDefault();
      send({ type: "key", key: e.key });
    }
  });

  // Direct paste support when the live browser viewport is focused
  window.addEventListener("paste", (e) => {
    if (document.activeElement !== viewportImg) return;
    const text = e.clipboardData?.getData("text");
    if (text) {
      e.preventDefault();
      send({ type: "type", text, press_enter: false });
    }
  });

  // Direct mouse wheel / trackpad scroll on live browser viewport
  let scrollTimer = null;
  viewportImg.addEventListener("wheel", (e) => {
    e.preventDefault();
    if (scrollTimer) return;
    scrollTimer = setTimeout(() => { scrollTimer = null; }, 90);
    send({ type: "scroll", dy: e.deltaY });
  }, { passive: false });

  connect();
})();
