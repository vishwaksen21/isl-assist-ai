# Backend (FastAPI)

Run the API:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `GET /api/session`
- `POST /api/predict/frame` (multipart form: `frame`, `session_id`)
- `POST /api/session/reset?session_id=...`

Model artifacts expected at repo-level:
- `artifacts/model.pt`
- `artifacts/labels.json`

Train them via `python scripts/train.py`.
