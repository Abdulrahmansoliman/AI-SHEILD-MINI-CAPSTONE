"""Check a running backend against the six original freeze fixtures."""
import argparse
import json
import math
from pathlib import Path

import httpx

from backend.detector import CHECKPOINT_SHA256, DEFAULT_THRESHOLD, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--artifacts", type=Path, default=Path(".local/backend/artifacts"))
    parser.add_argument("--fixtures", type=Path, default=Path(".local/backend/fixtures"))
    parser.add_argument("--output", type=Path, default=Path(".local/backend/backend_verification.json"))
    args = parser.parse_args()
    config = json.loads((args.artifacts / "detector_config.json").read_text())
    policy = config["comparison"]
    rows = []
    with httpx.Client(base_url=args.url, timeout=120, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        identity = health.json()["detector"]
        if identity["checkpoint_sha256"] != CHECKPOINT_SHA256 or identity["threshold"] != DEFAULT_THRESHOLD:
            raise RuntimeError("Running backend is not the frozen CF detector.")
        for fixture in config["fixtures"]:
            path = args.fixtures / Path(fixture["path"]).name
            if sha256_file(path) != fixture["sha256"]:
                raise RuntimeError(f"Fixture hash differs: {path.name}")
            with path.open("rb") as source:
                response = client.post("/analyze", files={"file": (path.name, source, "image/jpeg")})
            response.raise_for_status()
            report = response.json()
            result = report["detection"]
            passed = (
                math.isclose(result["ai_score"], fixture["expected_prob_fake"], rel_tol=policy["prob_rtol"], abs_tol=policy["prob_atol"])
                and math.isclose(result["logit"], fixture["expected_logit"], rel_tol=policy["logit_rtol"], abs_tol=policy["logit_atol"])
                and result["class_index"] == int(fixture["expected_prob_fake"] >= DEFAULT_THRESHOLD)
                and report["evidence"]["file"]["sha256"] == fixture["sha256"]
                and report["evidence"]["interpretation"]["detection_score_modified"] is False
                and report["evidence"]["metadata"]["status"] in {"absent", "present"}
                and report["evidence"]["provenance"]["status"] not in {"error", "unavailable", "unsupported"}
            )
            rows.append({
                "sample_id": fixture["sample_id"], "passed": passed,
                "expected_ai_score": fixture["expected_prob_fake"], "actual_ai_score": result["ai_score"],
                "prob_abs_error": abs(result["ai_score"] - fixture["expected_prob_fake"]),
                "logit_abs_error": abs(result["logit"] - fixture["expected_logit"]),
                "metadata_status": report["evidence"]["metadata"]["status"],
                "provenance_status": report["evidence"]["provenance"]["status"],
                "timing_seconds": report["timing_seconds"],
            })
    report = {
        "status": "PASS" if len(rows) == 6 and all(row["passed"] for row in rows) else "FAIL",
        "checked": len(rows), "detector": identity, "results": rows,
        "scope": "HTTP integration and freeze parity; not a generalization benchmark.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {sum(row['passed'] for row in rows)}/{len(rows)} HTTP fixture checks")
    print(f"Report: {args.output}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
