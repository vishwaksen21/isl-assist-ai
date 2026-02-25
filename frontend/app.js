const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const overlay = document.getElementById('overlay');
const backendUrlEl = document.getElementById('backendUrl');

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const clearBtn = document.getElementById('clearBtn');
const speakBtn = document.getElementById('speakBtn');

const predEl = document.getElementById('pred');
const confEl = document.getElementById('conf');
const statusEl = document.getElementById('status');
const outputEl = document.getElementById('output');
const correctLabelEl = document.getElementById('correctLabel');
const sendFeedbackBtn = document.getElementById('sendFeedbackBtn');
const resetBufferBtn = document.getElementById('resetBufferBtn');
const showLandmarksEl = document.getElementById('showLandmarks');

const labelGuideTextEl = document.getElementById('labelGuideText');
const labelGuideAnimEl = document.getElementById('labelGuideAnim');

// Mirror the camera for a natural "selfie" view.
// Important: we mirror both the preview (CSS) and the captured frames (canvas)
// so the backend sees the same orientation the user sees.
const MIRROR_VIDEO = true;

// MediaPipe Hands landmark graph (same as backend ST-GCN topology)
const HAND_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
];

function ensureOverlaySize() {
  const w = video.videoWidth || 0;
  const h = video.videoHeight || 0;
  if (!w || !h) return;
  if (overlay.width !== w) overlay.width = w;
  if (overlay.height !== h) overlay.height = h;
}

function clearOverlay() {
  const ctx = overlay.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawLandmarks(landmarks) {
  if (!showLandmarksEl || !showLandmarksEl.checked) return;
  if (!landmarks || !Array.isArray(landmarks) || landmarks.length !== 21) return;

  ensureOverlaySize();
  const w = overlay.width;
  const h = overlay.height;
  if (!w || !h) return;

  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  const pts = landmarks.map(p => ({ x: p[0] * w, y: p[1] * h }));
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }

  // bbox
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  ctx.lineWidth = 2;
  const pad = 10;
  ctx.strokeRect(minX - pad, minY - pad, (maxX - minX) + pad * 2, (maxY - minY) + pad * 2);

  // edges
  ctx.strokeStyle = 'rgba(80,160,255,0.85)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const [a, b] of HAND_EDGES) {
    ctx.moveTo(pts[a].x, pts[a].y);
    ctx.lineTo(pts[b].x, pts[b].y);
  }
  ctx.stroke();

  // points
  ctx.fillStyle = 'rgba(255,255,255,0.95)';
  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function updateLabelGuide() {
  const raw = (correctLabelEl?.value || '').trim();
  const label = raw.toUpperCase();

  if (!label) {
    labelGuideTextEl.textContent = 'Enter a label to see a cue.';
    labelGuideAnimEl.classList.add('is-hidden');
    labelGuideAnimEl.classList.remove('cue-wave', 'cue-nod', 'cue-pinch');
    return;
  }

  labelGuideAnimEl.classList.remove('cue-wave', 'cue-nod', 'cue-pinch');

  if (label === 'HELLO') {
    labelGuideTextEl.textContent = 'Cue: HELLO (wave side-to-side).';
    labelGuideAnimEl.classList.remove('is-hidden');
    labelGuideAnimEl.classList.add('cue-wave');
    return;
  }
  if (label === 'YES') {
    labelGuideTextEl.textContent = 'Cue: YES (small nod motion).';
    labelGuideAnimEl.classList.remove('is-hidden');
    labelGuideAnimEl.classList.add('cue-nod');
    return;
  }
  if (label === 'THANK_YOU') {
    labelGuideTextEl.textContent = 'Cue: THANK_YOU (pinch/close gesture).';
    labelGuideAnimEl.classList.remove('is-hidden');
    labelGuideAnimEl.classList.add('cue-pinch');
    return;
  }

  labelGuideTextEl.textContent = `Cue: ${label} (record a consistent gesture).`;
  labelGuideAnimEl.classList.add('is-hidden');
}

let stream = null;
let timer = null;
let sessionId = null;
let lastWord = '';
let lastPrediction = null;

// Stability window for isolated-sign recognition
const STABILITY_WINDOW = 8;
const STABILITY_REQUIRED = 6;
const CONF_THRESHOLD = 0.60;
let predHistory = [];

// Simple sentence formation: append tokens separated by space.
function appendWord(word, confidence) {
  if (!word) return;
  if (confidence < CONF_THRESHOLD) return;
  if (word === lastWord) return; // duplicate filtering

  const existing = outputEl.textContent.trim();
  const next = existing ? `${existing} ${word}` : word;
  outputEl.textContent = next;
  lastWord = word;
}

function getStablePrediction() {
  // Use only confident predictions in the window
  const confident = predHistory.filter(p => p.conf >= CONF_THRESHOLD);
  if (confident.length < STABILITY_REQUIRED) return null;

  const counts = new Map();
  for (const p of confident) {
    counts.set(p.pred, (counts.get(p.pred) || 0) + 1);
  }

  let bestPred = null;
  let bestCount = 0;
  for (const [pred, c] of counts.entries()) {
    if (c > bestCount) {
      bestPred = pred;
      bestCount = c;
    }
  }
  if (bestCount < STABILITY_REQUIRED) return null;

  const bestItems = confident.filter(p => p.pred === bestPred);
  const avgConf = bestItems.reduce((s, p) => s + p.conf, 0) / bestItems.length;
  return { pred: bestPred, conf: avgConf };
}

function setStatus(s) {
  statusEl.textContent = s;
}

async function createSession() {
  const base = backendUrlEl.value.replace(/\/$/, '');
  const res = await fetch(`${base}/api/session`);
  if (!res.ok) throw new Error('Failed to create session');
  const data = await res.json();
  return data.session_id;
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Camera API not available (getUserMedia). Open the frontend using http://localhost:5173 (not file:// and not a remote IP). Use Chrome/Edge and allow camera permission.'
    );
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  } catch (e) {
    const name = e && typeof e === 'object' && 'name' in e ? e.name : '';
    if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
      throw new Error('Camera permission denied. Allow camera access in the browser and reload.');
    }
    if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
      throw new Error('No camera found. Connect/enable a webcam and reload.');
    }
    throw e;
  }
  video.srcObject = stream;
  await video.play();
  ensureOverlaySize();
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  clearOverlay();
}

function frameToBlob() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (MIRROR_VIDEO) {
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();
  } else {
    ctx.drawImage(video, 0, 0, w, h);
  }

  return new Promise(resolve => {
    canvas.toBlob(blob => resolve(blob), 'image/jpeg', 0.8);
  });
}

async function sendFrame() {
  const base = backendUrlEl.value.replace(/\/$/, '');
  const debug = showLandmarksEl && showLandmarksEl.checked ? 1 : 0;

  const blob = await frameToBlob();
  if (!blob) return;

  const form = new FormData();
  form.append('frame', blob, 'frame.jpg');
  form.append('session_id', sessionId);

  const res = await fetch(`${base}/api/predict/frame?debug=${debug}`, { method: 'POST', body: form });
  if (!res.ok) {
    setStatus(`Backend error (${res.status})`);
    predEl.textContent = '-';
    confEl.textContent = '-';
    clearOverlay();
    return;
  }

  const data = await res.json();

  if (data.landmarks) {
    drawLandmarks(data.landmarks);
  } else {
    clearOverlay();
  }

  if (!data.detected) {
    predEl.textContent = '-';
    confEl.textContent = '-';
    setStatus('No hand detected');
    return;
  }

  if (!data.ready) {
    setStatus(`Buffering frames (${data.buffer_size}/30)`);
    return;
  }

  const pred = data.prediction;
  const conf = data.confidence;

  lastPrediction = { pred, conf };

  predHistory.push({ pred, conf });
  if (predHistory.length > STABILITY_WINDOW) predHistory.shift();

  predEl.textContent = pred;
  confEl.textContent = `${(conf * 100).toFixed(1)}%`;
  setStatus('Running');

  const stable = getStablePrediction();
  if (stable) {
    appendWord(stable.pred, stable.conf);
  }
}

async function sendFeedback() {
  const base = backendUrlEl.value.replace(/\/$/, '');
  const correctLabel = (correctLabelEl.value || '').trim();
  if (!sessionId) {
    alert('Start first to create a session.');
    return;
  }
  if (!correctLabel) {
    alert('Enter a correct label first.');
    return;
  }

  try {
    setStatus('Saving sample...');
    const res = await fetch(`${base}/api/dataset/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, label: correctLabel }),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt);
    }
    await res.json();
    setStatus('Sample saved');
  } catch (e) {
    console.error(e);
    setStatus('Save failed');
  }
}

async function resetBuffer() {
  const base = backendUrlEl.value.replace(/\/$/, '');
  if (!sessionId) {
    alert('Start first to create a session.');
    return;
  }
  try {
    setStatus('Resetting buffer...');
    const res = await fetch(`${base}/api/session/reset?session_id=${encodeURIComponent(sessionId)}`, {
      method: 'POST',
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt);
    }
    predHistory = [];
    lastWord = '';
    lastPrediction = null;
    predEl.textContent = '-';
    confEl.textContent = '-';
    clearOverlay();
    setStatus('Buffer reset');
  } catch (e) {
    console.error(e);
    setStatus('Reset failed');
  }
}

async function start() {
  startBtn.disabled = true;
  try {
    setStatus('Starting camera...');
    await startCamera();

    setStatus('Creating session...');
    sessionId = await createSession();

    // ~10 FPS
    timer = setInterval(() => {
      sendFrame().catch(err => {
        console.error(err);
        predEl.textContent = '-';
        confEl.textContent = '-';
        clearOverlay();
        setStatus('Error sending frame');
      });
    }, 100);

    stopBtn.disabled = false;
    setStatus('Running');
  } catch (e) {
    console.error(e);
    setStatus(e.message || 'Failed to start');
    startBtn.disabled = false;
  }
}

function stop() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
  stopCamera();
  stopBtn.disabled = true;
  startBtn.disabled = false;
  setStatus('Stopped');
}

function clearText() {
  outputEl.textContent = '';
  lastWord = '';
  predHistory = [];
}

function speak() {
  const text = outputEl.textContent.trim();
  if (!text) return;

  if (!('speechSynthesis' in window)) {
    alert('Web Speech API is not supported in this browser.');
    return;
  }

  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.0;
  u.pitch = 1.0;
  window.speechSynthesis.speak(u);
}

startBtn.addEventListener('click', () => start());
stopBtn.addEventListener('click', () => stop());
clearBtn.addEventListener('click', () => clearText());
speakBtn.addEventListener('click', () => speak());
sendFeedbackBtn.addEventListener('click', () => sendFeedback());
resetBufferBtn.addEventListener('click', () => resetBuffer());

correctLabelEl.addEventListener('input', () => updateLabelGuide());
showLandmarksEl.addEventListener('change', () => {
  if (!showLandmarksEl.checked) clearOverlay();
});

updateLabelGuide();

setStatus('Idle');
