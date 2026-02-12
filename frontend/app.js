const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
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
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  await video.play();
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
}

function frameToBlob() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);

  return new Promise(resolve => {
    canvas.toBlob(blob => resolve(blob), 'image/jpeg', 0.8);
  });
}

async function sendFrame() {
  const base = backendUrlEl.value.replace(/\/$/, '');

  const blob = await frameToBlob();
  if (!blob) return;

  const form = new FormData();
  form.append('frame', blob, 'frame.jpg');
  form.append('session_id', sessionId);

  const res = await fetch(`${base}/api/predict/frame`, { method: 'POST', body: form });
  if (!res.ok) {
    setStatus(`Backend error (${res.status})`);
    return;
  }

  const data = await res.json();

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

setStatus('Idle');
