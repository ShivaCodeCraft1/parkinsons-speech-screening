// Plain JS front end. Talks only to the documented API (/health, /model-info, /predict, /analyze, /demo/*).
// API base can be overridden: <script>window.PD_API_BASE="http://localhost:8000"</script> before app.js.
(() => {
  const API = (window.PD_API_BASE || "").replace(/\/$/, "");
  const MAX_BYTES = 20_000_000;
  const $ = (id) => document.getElementById(id);
  let currentFile = null;     // File/Blob that will be uploaded (WAV when we could convert it)
  let recorder = null, chunks = [], stream = null, timerId = null, t0 = 0, modelLoaded = false;

  const show = (el, on = true) => { el.hidden = !on; };
  const setError = (msg) => { $("input-error").textContent = msg || ""; show($("input-error"), !!msg); };
  const fmtTime = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const el = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };

  // ---------- model status ----------
  async function loadStatus() {
    try {
      const h = await (await fetch(`${API}/health`)).json();
      modelLoaded = !!h.model_loaded;
      if (modelLoaded) {
        const info = await (await fetch(`${API}/model-info`)).json();
        const ev = info.evaluation_status || "";
        $("model-status").textContent = `Model ${info.model_name} (${info.model_version}). ${ev}`;
        if (info.task_families) $("task-hint").textContent = `This model was trained on task(s): ${info.task_families.join(", ")}. For sustained vowels, hold a steady "aaah" for about 5 seconds.`;
      } else {
        $("model-status").textContent = "No trained model found: analysis (validation, plots, features) still works, but prediction is unavailable. Train a model first (see README).";
      }
    } catch (e) {
      $("model-status").textContent = "Cannot reach the API. Is the server running?";
    }
    try {
      const r = await fetch(`${API}/demo/samples?partition=test`);
      if (r.ok) {
        const items = await r.json();
        if (items.length) {
          const sel = $("demo-select");
          items.forEach((s) => { const o = el("option", `${s.sample_id} (${s.task_code || "?"}, ${s.partition})`); o.value = s.sample_id; sel.appendChild(o); });
          show($("demo-card"));
        }
      }
    } catch (_) { /* demo mode is optional */ }
  }

  // ---------- WAV conversion ----------
  function encodeWav(float32, sampleRate) {
    const buf = new ArrayBuffer(44 + float32.length * 2), v = new DataView(buf);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    w(0, "RIFF"); v.setUint32(4, 36 + float32.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    w(36, "data"); v.setUint32(40, float32.length * 2, true);
    for (let i = 0; i < float32.length; i++) { const s = Math.max(-1, Math.min(1, float32[i])); v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true); }
    return new Blob([buf], { type: "audio/wav" });
  }
  async function toWav(blob) {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    try {
      const audio = await ctx.decodeAudioData(await blob.arrayBuffer());
      const ch = audio.numberOfChannels, n = audio.length, mono = new Float32Array(n);
      for (let c = 0; c < ch; c++) { const d = audio.getChannelData(c); for (let i = 0; i < n; i++) mono[i] += d[i] / ch; }
      return encodeWav(mono, audio.sampleRate);
    } finally { ctx.close(); }
  }

  function setCurrent(blob, name) {
    currentFile = new File([blob], name, { type: blob.type || "audio/wav" });
    const url = URL.createObjectURL(blob);
    $("player").src = url;
    $("client-info").textContent = `${name} - ${(blob.size / 1000).toFixed(0)} kB (${blob.type || "unknown type"})`;
    show($("playback")); show($("results"), false);
  }

  // ---------- recording ----------
  $("btn-start").onclick = async () => {
    setError("");
    if (!navigator.mediaDevices || !window.MediaRecorder) { setError("This browser cannot record audio. Use the upload option or a recent Chrome/Edge/Firefox."); return; }
    try {
      // Browser audio "enhancements" change jitter/shimmer/HNR, so switch them off.
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      setError(e.name === "NotAllowedError" ? "Microphone permission was denied. Allow microphone access in the browser address bar and try again." : `Could not open the microphone: ${e.message}`);
      return;
    }
    chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(timerId);
      const raw = new Blob(chunks, { type: recorder.mimeType });
      try { setCurrent(await toWav(raw), "recording.wav"); }
      catch (e) {
  console.error("[rec] decoding failed:", e);
  setError(`Recording failed: ${e.name}: ${e.message}`);
}
    };
    recorder.start();
    t0 = Date.now();
    timerId = setInterval(() => { $("timer").textContent = fmtTime((Date.now() - t0) / 1000); }, 200);
    $("btn-start").disabled = true; $("btn-stop").disabled = false;
  };
  $("btn-stop").onclick = () => { if (recorder && recorder.state !== "inactive") recorder.stop(); $("btn-start").disabled = false; $("btn-stop").disabled = true; };

  // ---------- upload ----------
  $("file-input").onchange = async (ev) => {
    setError("");
    const f = ev.target.files[0];
    if (!f) return;
    if (f.size > MAX_BYTES) { setError("File is larger than 20 MB."); return; }
    if (/\.wav$/i.test(f.name) || f.type === "audio/wav" || f.type === "audio/x-wav") { setCurrent(f, f.name); return; }
    try { setCurrent(await toWav(f), f.name.replace(/\.[^.]+$/, "") + ".wav"); }
    catch (e) {
  console.error("[upload] conversion failed:", e);
  setCurrent(f, f.name);
  setError("This browser could not convert the file to WAV; the server will try to decode it (may fail without ffmpeg).");
}
  };

  // ---------- analysis ----------
  async function postFile(path) {
    const fd = new FormData(); fd.append("file", currentFile, currentFile.name);
    const r = await fetch(`${API}${path}`, { method: "POST", body: fd });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.error ? `${body.error.message} (${body.error.code})` : (body.detail ? JSON.stringify(body.detail) : `HTTP ${r.status}`));
    return body;
  }
  $("btn-analyze").onclick = async () => {
    setError(""); show($("busy")); $("btn-analyze").disabled = true;
    try {
      const explain = $("chk-explain").checked;
      const body = modelLoaded ? await postFile(`/predict?explain=${explain}`) : { analysis: await postFile("/analyze"), result: null };
      render(body.analysis, body.result, body.disclaimer);
    } catch (e) { setError(e.message); show($("results"), false); }
    finally { show($("busy"), false); $("btn-analyze").disabled = false; }
  };

  function renderResult(box, r) {
    box.replaceChildren();
    const wrap = el("div", undefined, "result");
    wrap.appendChild(el("div", r.prediction_label, "big"));
    wrap.appendChild(el("div", `Model score: ${r.score.toFixed(4)} (uncalibrated - NOT a probability). Operating threshold: ${r.threshold.toFixed(4)}.`));
    wrap.appendChild(el("div", `Model: ${r.model_name} - version ${r.model_version}`, "muted"));
    if (r.explanation) {
      wrap.appendChild(el("h3", `Top contributing features (${r.explanation.method})`));
      const ul = el("ul"); r.explanation.top_features.forEach((f) => ul.appendChild(el("li", `${f.feature}: ${f.contribution >= 0 ? "+" : ""}${f.contribution.toFixed(3)}`)));
      wrap.appendChild(ul); wrap.appendChild(el("div", r.explanation.note, "muted"));
    }
    box.appendChild(wrap);
    return r.warnings;
  }

  function render(a, result, disclaimer) {
    show($("results"));
    const box = $("prediction-box");
    let warnings = a.validation.warnings;
    if (result) warnings = renderResult(box, result);
    else { box.replaceChildren(el("div", "No trained model is loaded, so no prediction is shown.", "warn")); }
    const wbox = $("warnings"); wbox.replaceChildren();
    warnings.forEach((w) => wbox.appendChild(el("div", w.message, "warn")));
    const dl = $("validation-info"); dl.replaceChildren();
    const info = a.validation.info;
    [["Validation", a.validation.ok ? "passed" : "failed"], ["Duration", `${info.duration_sec} s`], ["Original sample rate", `${info.original_sample_rate} Hz`],
     ["Analysed at", `${a.sample_rate} Hz`], ["Source channels", info.source_channels], ["Peak / RMS", `${info.peak_abs} / ${info.rms_dbfs} dBFS`],
     ["Clipped samples", info.clip_fraction ?? "n/a"]].forEach(([k, v]) => { dl.appendChild(el("dt", k)); dl.appendChild(el("dd", String(v))); });
    drawWave(a.waveform); drawSpec(a.spectrogram);
    const tb = document.querySelector("#features tbody"); tb.replaceChildren();
    Object.entries(a.features).forEach(([k, v]) => { const tr = el("tr"); tr.appendChild(el("td", k)); tr.appendChild(el("td", v === null ? "undefined" : Number(v).toPrecision(5))); tb.appendChild(tr); });
    $("disclaimer").textContent = disclaimer || "Research prototype only. Not a medical device and not a diagnosis.";
  }

  function drawWave(w) {
    const c = $("wave"), g = c.getContext("2d"); g.clearRect(0, 0, c.width, c.height);
    if (!w) return;
    const n = w.min.length, mid = c.height / 2; g.strokeStyle = "#2457c5"; g.beginPath();
    for (let i = 0; i < n; i++) { const x = (i / n) * c.width; g.moveTo(x, mid - w.max[i] * mid); g.lineTo(x, mid - w.min[i] * mid); }
    g.stroke();
  }
  function drawSpec(s) {
    const c = $("spec"), g = c.getContext("2d"); g.clearRect(0, 0, c.width, c.height);
    if (!s) return;
    const nF = s.db.length, nT = s.db[0].length, cw = c.width / nT, ch = c.height / nF;
    for (let f = 0; f < nF; f++) for (let t = 0; t < nT; t++) {
      const v = Math.min(1, Math.max(0, (s.db[f][t] + 80) / 80)), shade = Math.round(255 * (1 - v));
      g.fillStyle = `rgb(${shade},${shade},${Math.min(255, shade + 40)})`; g.fillRect(t * cw, c.height - (f + 1) * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }

  // ---------- demo mode ----------
  $("btn-demo").onclick = async () => {
    const id = $("demo-select").value, out = $("demo-result"); out.replaceChildren(el("p", "Running..."));
    try {
      const r = await fetch(`${API}/demo/predict/${encodeURIComponent(id)}?explain=false`, { method: "POST" });
      const b = await r.json(); if (!r.ok) throw new Error(b.detail || (b.error && b.error.message) || `HTTP ${r.status}`);
      out.replaceChildren(); const box = el("div"); out.appendChild(box); renderResult(box, b.result);
      const gt = el("div", undefined, "result");
      gt.appendChild(el("h3", "Dataset label (ground truth - NOT a model output)"));
      gt.appendChild(el("div", `${b.dataset_label_not_a_model_output} - this recording's subject is in the "${b.partition}" partition.`));
      out.appendChild(gt); render(b.analysis, b.result, b.disclaimer);
    } catch (e) { out.replaceChildren(el("p", e.message, "error")); }
  };

  loadStatus();
})();
