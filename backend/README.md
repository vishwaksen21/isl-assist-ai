# Backend (FastAPI)

Run the API:

```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Note (macOS / Homebrew Python): if you see errors mentioning **NumPy 2.x** and
MediaPipe/TensorFlow, you are likely running `uvicorn` from a different Python
installation (e.g. Homebrew Python 3.12). Always start with `python -m uvicorn`
from the same environment you used to install `requirements.txt`.

Endpoints:
- `GET /health`
- `GET /api/session`
- `POST /api/predict/frame` (multipart form: `frame`, `session_id`)
- `POST /api/session/reset?session_id=...`

Model artifacts expected at repo-level:
- `artifacts/model.pt`
- `artifacts/labels.json`

Train them via `python scripts/train.py`.
