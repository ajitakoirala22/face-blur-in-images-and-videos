from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from face_detector import FaceDetector
from pipeline import process_image, process_video


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024

_detector_cache = {}
_jobs = {}
_jobs_lock = threading.Lock()


def get_detector(device: str) -> FaceDetector:
    key = device.lower()
    if key not in _detector_cache:
        _detector_cache[key] = FaceDetector(device=key)
        print(f"Using device: {_detector_cache[key].device}")
    return _detector_cache[key]


def is_video(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".mov", ".avi"}


def _set_job(job_id: str, **fields):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _run_job(
    job_id: str,
    input_path: Path,
    output_path: Path,
    conf: float,
    enhance: bool,
    device: str,
):
    try:
        _set_job(job_id, status="running", progress=2, message="Initializing detector")
        detector = get_detector(device)
        _set_job(job_id, device_used=detector.device)

        def progress_cb(progress: int, message: str):
            _set_job(job_id, progress=progress, message=message)

        if is_video(input_path):
            process_video(
                str(input_path),
                str(output_path),
                detector,
                conf_thresh=conf,
                enhance=enhance,
                progress_cb=progress_cb,
            )
            media_type = "video"
        else:
            process_image(
                str(input_path),
                str(output_path),
                detector,
                conf_thresh=conf,
                enhance=enhance,
                progress_cb=progress_cb,
            )
            media_type = "image"

        _set_job(
            job_id,
            status="completed",
            progress=100,
            message="Completed",
            type=media_type,
            output_url=f"/media/{output_path.name}",
            download_name=output_path.name,
        )
    except Exception as exc:
        _set_job(job_id, status="failed", message=f"Processing failed: {exc}")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/process")
def process_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    safe_name = secure_filename(f.filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": "Unsupported file type"}), 400

    conf = float(request.form.get("confidence", "0.20"))
    conf = min(max(conf, 0.1), 0.95)
    enhance = request.form.get("enhance", "false").lower() == "true"
    device = request.form.get("device", "auto").lower()

    uid = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{uid}{ext}"
    output_ext = ".mp4" if ext in {".mp4", ".mov", ".avi"} else ext
    output_path = OUTPUT_DIR / f"{uid}_blurred{output_ext}"
    f.save(str(input_path))

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Queued",
            "type": "video" if is_video(input_path) else "image",
            "output_url": None,
            "download_name": None,
            "device_used": None,
        }

    worker = threading.Thread(
        target=_run_job,
        args=(job_id, input_path, output_path, conf, enhance, device),
        daemon=True,
    )
    worker.start()

    return jsonify({"ok": True, "job_id": job_id})


@app.get("/status/<job_id>")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(job)


@app.get("/media/<path:filename>")
def media(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
