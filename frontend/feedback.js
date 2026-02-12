export async function sendFeedback(backendBaseUrl, sessionId, correctLabel) {
  const base = backendBaseUrl.replace(/\/$/, '');
  const res = await fetch(`${base}/api/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, correct_label: correctLabel }),
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Feedback failed (${res.status}): ${txt}`);
  }
  return res.json();
}
