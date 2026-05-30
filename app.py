import re
import uuid
import shutil
import subprocess
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

RESULTS_DIR = Path("static/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store: job_id -> {status, stage, error, piano_url, orchestra_url}
jobs: dict[str, dict] = {}

ORCHESTRA_STEMS = ["bass", "drums", "guitar", "other", "vocals"]


def _run(cmd: list[str], capture: bool = True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, check=True, text=True,
        capture_output=capture,
        **kwargs
    )


_PERCENT_RE = re.compile(r"(\d+)%")


def _run_with_progress(cmd: list[str], on_progress) -> None:
    """Run a command, parsing tqdm-style '..%' progress from stderr.

    demucs writes a tqdm bar to stderr using carriage returns. We read it in
    small chunks, split on \\r / \\n, and report the latest percentage seen.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    buf = ""
    last_err_line = ""
    assert proc.stderr is not None
    # Read char-by-char so carriage-return-updated progress bars are caught.
    while True:
        ch = proc.stderr.read(1)
        if not ch:
            break
        if ch in ("\r", "\n"):
            line = buf.strip()
            buf = ""
            if not line:
                continue
            last_err_line = line
            m = _PERCENT_RE.search(line)
            if m:
                on_progress(int(m.group(1)))
        else:
            buf += ch

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, stderr=last_err_line
        )


def process_job(job_id: str, youtube_url: str):
    job = jobs[job_id]
    work_dir = TMP_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Download audio
        job["stage"] = "Downloading audio..."
        audio_path = work_dir / "input.%(ext)s"
        _run([
            "venv/bin/yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "-o", str(audio_path),
            youtube_url,
        ])

        input_wav = next(work_dir.glob("input.*"))

        # 2. Separate stems with demucs htdemucs_6s
        # --mp3 uses lameenc to save stems, avoiding the torchaudio/torchcodec issue
        job["stage"] = "Separating stems (this takes a few minutes)..."
        job["progress"] = 0
        _run_with_progress(
            [
                "venv/bin/python", "-m", "demucs",
                "-n", "htdemucs_6s",
                "-o", str(work_dir / "demucs_out"),
                "--segment", "7",
                "--shifts", "1",
                "--mp3",
                str(input_wav),
            ],
            on_progress=lambda pct: job.__setitem__("progress", pct),
        )
        job["progress"] = 100

        stems_dir = work_dir / "demucs_out" / "htdemucs_6s" / input_wav.stem

        # 3. Mix non-piano stems into orchestra track using ffmpeg
        job["stage"] = "Mixing orchestra track..."
        job["progress"] = None
        out_dir = RESULTS_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        orchestra_path = out_dir / "orchestra.mp3"
        piano_path = out_dir / "piano.mp3"

        stems_to_mix = [
            str(stems_dir / f"{s}.mp3")
            for s in ORCHESTRA_STEMS
            if (stems_dir / f"{s}.mp3").exists()
        ]
        n = len(stems_to_mix)
        ffmpeg_inputs = []
        for s in stems_to_mix:
            ffmpeg_inputs += ["-i", s]

        _run([
            "ffmpeg", "-y",
            *ffmpeg_inputs,
            "-filter_complex", f"amix=inputs={n}:normalize=0",
            "-ac", "2", "-q:a", "2",
            str(orchestra_path),
        ])

        # Piano stem is already mp3 — copy it to results
        shutil.copy(stems_dir / "piano.mp3", piano_path)

        job["orchestra_url"] = f"/static/results/{job_id}/orchestra.mp3"
        job["piano_url"] = f"/static/results/{job_id}/piano.mp3"
        job["status"] = "done"
        job["stage"] = "Done"

    except subprocess.CalledProcessError as e:
        job["status"] = "error"
        # e.stderr may contain tqdm progress garbage; take only the last meaningful line
        stderr_lines = [l.strip() for l in (e.stderr or "").splitlines() if l.strip() and "%" not in l]
        job["error"] = stderr_lines[-1] if stderr_lines else f"Command failed: {e.cmd[0]}"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()
    youtube_url = data.get("url", "").strip()

    if not youtube_url or ("youtube.com" not in youtube_url and "youtu.be" not in youtube_url):
        return jsonify({"error": "Please provide a valid YouTube URL"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "stage": "Starting...", "error": None, "progress": None}

    thread = threading.Thread(target=process_job, args=(job_id, youtube_url), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


if __name__ == "__main__":
    app.run(debug=True)
