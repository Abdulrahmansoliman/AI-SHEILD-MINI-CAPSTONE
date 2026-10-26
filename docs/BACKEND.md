# CF and Provenance Backend

The standalone FastAPI backend runs the verified fine-tuned Community Forensics
checkpoint and recorded-provenance module on the same original upload. It
returns both results without metadata changing CF's score. The old app remains
available separately.

## Local Start

This workspace already has isolated CPU dependencies and verified artifacts
under the ignored `.local/` directory. Start from the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_backend.ps1 -Port 8000
```

- API documentation and upload form: http://127.0.0.1:8000/docs
- Readiness and detector identity: http://127.0.0.1:8000/health
- Upload endpoint: `POST /analyze`, multipart file field named `file`
- Optional GPS extraction: `POST /analyze?include_gps=true`

Example from Windows:

```powershell
curl.exe -F "file=@image.jpg" http://127.0.0.1:8000/analyze
```

## Fresh Environment

Use Python 3.11. Install CPU PyTorch first, then the backend dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-backend.txt
```

Install ExifTool as described in [PROVENANCE.md](PROVENANCE.md). The dedicated
backend requirements intentionally avoid the previous TensorFlow/fusion stack.
Do not install the legacy root `requirements.txt` into this environment.

Place these original verified artifacts in `.local/backend/artifacts/`, or set
`AI_SHIELD_ARTIFACT_DIR` to their directory:

- `best.pt`: Drive file `1rE9pc32i-BCijkQ0kJdqQJC9ioRi3RzZ`
- `detector_config.json`: Drive file `1WqSp31ChyGAMSRE8QiqwKaKOepn7Ax6c`
- `verification_report.json`: Drive file `1YBb2p93INzg6VH9o5M5tHMeM5eZibGKn`

These artifacts came from the passing freeze run
`20261013T143008Z_5bce09a8`. The unrelated `best.pt` in the repository root is not
the selected CF checkpoint. Model weights and downloaded image fixtures remain
outside Git.

For Linux or a standard virtual environment:

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## Frozen Detector

The repository's `src/ai_shield_cf_inference.py` is byte-identical to the verified
Colab export. The backend imports this local module rather than executing code
downloaded from Drive. Startup checks source, checkpoint and verification-report
hashes, the selected threshold, label mapping, model revision, preprocessing,
and compatible runtime versions.

- Checkpoint SHA-256: `51532d28870706a3c380a29889f634c556583bd62ed74775e5e610d5fd9d7b5f`
- Default threshold: `0.6339887976646423` (validation max-F1)
- Frame 0, EXIF orientation, RGB, shorter-side resize 440 bilinear with antialias,
  center crop 384, ImageNet normalization, float32
- Label 0: real; label 1: generated; decision uses sigmoid score >= threshold

The API calls its output `ai_score`; calibration is not established simply by
applying sigmoid. `likely_not_ai_generated` is a detector classification, not
proof of an authentic camera photograph.

## Response

```json
{
  "schema_version": "1.0",
  "detection": {
    "ai_score": 0.94,
    "class_index": 1,
    "label": "likely_ai_generated",
    "threshold": 0.6339887976646423,
    "detector_id": "cf_finetune_v1_max_f1"
  },
  "evidence": {
    "metadata": {"status": "absent"},
    "provenance": {"status": "absent"},
    "interpretation": {"origin": "unknown", "detection_score_modified": false}
  },
  "timing_seconds": {"detection": 0.3, "evidence": 0.2, "total": 0.6}
}
```

The example is abbreviated; real responses include detector identity, runtime
versions, file hash, recorded fields, credential claims, and validation details.
Raw metadata/manifests and internal processing-error messages are omitted from
the API response. The local provenance CLI remains available for raw inspection.

## Upload Behavior

- Supports decoded JPEG, PNG, and WebP, regardless of uploaded name/MIME.
- Limits files to 25 MiB and images to 20 million pixels; request-body size is
  also bounded while streaming, including multipart overhead.
- Keeps original bytes intact and processes provenance before CF preprocessing.
- Allows one analysis at a time per server process; concurrent attempts return
  HTTP 429 with a retry delay.
- Deletes temporary uploads after successful or failed analysis.
- Returns evidence errors alongside a valid detection when evidence extraction
  fails; detector errors return HTTP 500 rather than a made-up classification.
- Returns HTTP 413 for size limits and HTTP 422 for invalid images.
- C2PA remains offline; missing credentials and remote references remain distinct.

## Configuration

| Environment variable | Default / purpose |
| --- | --- |
| `AI_SHIELD_ARTIFACT_DIR` | `.local/backend/artifacts` |
| `AI_SHIELD_DEVICE` | `cpu` |
| `AI_SHIELD_TORCH_THREADS` | `2` |
| `EXIFTOOL_PATH` | ExifTool executable; Windows launcher detects the bundled copy |
| `AI_SHIELD_TRUST_ANCHORS` | Optional PEM anchors, chosen independently of uploads |
| `AI_SHIELD_CORS_ORIGINS` | Comma-separated frontend origins; defaults to localhost:3000 |

## Tests

Install `httpx==0.28.1` for FastAPI's test client, then run:

```powershell
python -m unittest discover -s tests -v
```

Enable actual checkpoint and saved-prediction checks with
`AI_SHIELD_CF_INTEGRATION=1`. Put the six original freeze fixtures in
`.local/backend/fixtures`, or set `AI_SHIELD_CF_FIXTURES`. Their names, hashes,
expected logits, probabilities and comparison tolerances come from the verified
configuration. CPU parity checks are implementation checks, not new accuracy
benchmarks. Provenance integration tests additionally use the fixture setup in
[PROVENANCE.md](PROVENANCE.md).

To verify a running API over actual HTTP against all six saved fixtures:

```powershell
python -m scripts.verify_cf_backend --url http://127.0.0.1:8000
```

This saves `.local/backend/backend_verification.json`, including score/logit
differences, evidence statuses, and measured request timings.

## Deployment Later

This is a local backend, not a public production deployment. The planned website
will send its original uploads to `/analyze` and display `detection` and `evidence`.
Before exposing it publicly, configure HTTPS and the actual frontend origin,
request timeouts/rate limits, and validate the complete environment on the chosen
server. Run one API worker initially to avoid duplicate model copies. C2PA's
native processing still needs a server-level timeout/resource boundary for public
untrusted uploads.
