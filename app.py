import os
import re
import uuid
import shutil
import subprocess
import threading
from functools import wraps
from pathlib import Path

import jwt
import requests as http_requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, g
from supabase import create_client

load_dotenv()

app = Flask(__name__)

# ── Supabase config ──
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_ANON_KEY = os.environ["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
SUPABASE_JWT_SECRET = os.environ["SUPABASE_JWT_SECRET"]
sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Fetch the JWKS (JSON Web Key Set) from Supabase for RS256/EdDSA token verification
_JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
try:
    _jwks_response = http_requests.get(_JWKS_URL, timeout=10)
    _jwks_data = _jwks_response.json()
    _jwks_client = jwt.PyJWKClient.__new__(jwt.PyJWKClient)
    # Manually set the cached keys
    _jwks_client._jwks_uri = _JWKS_URL
    # Actually just use the proper constructor
    _jwks_client = jwt.PyJWKClient(_JWKS_URL)
    print(f"[AUTH] Loaded JWKS from {_JWKS_URL}")
except Exception as e:
    _jwks_client = None
    print(f"[AUTH] Could not load JWKS: {e}. Falling back to SUPABASE_JWT_SECRET.")

# Expose public keys to Jinja templates (for Supabase JS client)
app.config["SUPABASE_URL"] = SUPABASE_URL
app.config["SUPABASE_ANON_KEY"] = SUPABASE_ANON_KEY

STORAGE_BUCKET = "extractions"

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store (for tracking in-progress jobs only)
jobs: dict[str, dict] = {}

ORCHESTRA_STEMS = ["bass", "drums", "guitar", "other", "vocals"]


# ── Auth helpers ──

def _get_user_id_from_token() -> str | None:
    """Extract user_id from the Authorization: Bearer <token> header.
    Returns None if no token or invalid token."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]
    try:
        header = jwt.get_unverified_header(token)
        alg = header.get("alg", "HS256")
        print(f"[AUTH] Token alg={alg}, kid={header.get('kid')}")

        if alg == "HS256":
            # Legacy: symmetric secret
            payload = jwt.decode(
                token, SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
            )
        else:
            # Newer Supabase: asymmetric key (EdDSA / RS256) — use JWKS
            if _jwks_client is None:
                print("[AUTH] No JWKS client available for asymmetric verification")
                return None
            signing_key = _jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token, signing_key.key,
                algorithms=[alg],
                audience="authenticated",
            )

        return payload.get("sub")
    except Exception as e:
        print(f"[AUTH] JWT decode failed: {type(e).__name__}: {e}")
        return None


def require_auth(f):
    """Decorator: reject if no valid auth token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user_id = _get_user_id_from_token()
        if not user_id:
            return jsonify({"error": "Authentication required"}), 401
        g.user_id = user_id
        return f(*args, **kwargs)
    return decorated


def optional_auth(f):
    """Decorator: set g.user_id if token present, None otherwise."""
    @wraps(f)
    def decorated(*args, **kwargs):
        g.user_id = _get_user_id_from_token()
        return f(*args, **kwargs)
    return decorated


# ── Processing helpers ──

def _run(cmd: list[str], capture: bool = True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, check=True, text=True,
        capture_output=capture,
        **kwargs
    )


_PERCENT_RE = re.compile(r"(\d+)%")


def _run_with_progress(cmd: list[str], on_progress) -> None:
    """Run a command, parsing tqdm-style '..%' progress from stderr."""
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


ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"}
MAX_UPLOAD_MB = 200

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# ── Supabase helpers ──

def _upload_to_storage(local_path: Path, storage_path: str) -> str:
    """Upload a local file to Supabase Storage and return its public URL."""
    with open(local_path, "rb") as f:
        sb.storage.from_(STORAGE_BUCKET).upload(
            storage_path,
            f,
            file_options={"content-type": "audio/mpeg", "upsert": "true"},
        )
    public_url = sb.storage.from_(STORAGE_BUCKET).get_public_url(storage_path)
    return public_url


def _save_extraction(job_id: str, source_type: str, source_label: str,
                     piano_path: str, orchestra_path: str,
                     user_id: str | None = None):
    """Insert extraction metadata into the Supabase database."""
    row = {
        "id": job_id,
        "source_type": source_type,
        "source_label": source_label,
        "piano_path": piano_path,
        "orchestra_path": orchestra_path,
        "status": "done",
    }
    if user_id:
        row["user_id"] = user_id
    sb.table("extractions").insert(row).execute()


def process_job(job_id: str, youtube_url: str | None = None,
                source_label: str | None = None,
                user_id: str | None = None):
    job = jobs[job_id]
    work_dir = TMP_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Obtain audio
        if youtube_url:
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
            src_type = "youtube"
            src_label = youtube_url
        else:
            input_wav = next(work_dir.glob("input.*"))
            src_type = "upload"
            src_label = source_label or input_wav.name

        # 2. Separate stems with demucs
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

        # 3. Mix non-piano stems into orchestra track
        job["stage"] = "Mixing orchestra track..."
        job["progress"] = None

        orchestra_local = work_dir / "orchestra.mp3"
        piano_local = stems_dir / "piano.mp3"

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
            str(orchestra_local),
        ])

        # 4. Upload results to Supabase Storage
        job["stage"] = "Saving results..."
        piano_storage = f"{job_id}/piano.mp3"
        orchestra_storage = f"{job_id}/orchestra.mp3"

        piano_url = _upload_to_storage(piano_local, piano_storage)
        orchestra_url = _upload_to_storage(orchestra_local, orchestra_storage)

        # 5. Save metadata to Supabase database
        _save_extraction(job_id, src_type, src_label,
                         piano_storage, orchestra_storage,
                         user_id=user_id)

        job["orchestra_url"] = orchestra_url
        job["piano_url"] = piano_url
        job["status"] = "done"
        job["stage"] = "Done"

        # 6. Clean up local tmp files
        shutil.rmtree(work_dir, ignore_errors=True)

    except subprocess.CalledProcessError as e:
        job["status"] = "error"
        stderr_lines = [l.strip() for l in (e.stderr or "").splitlines()
                        if l.strip() and "%" not in l]
        job["error"] = stderr_lines[-1] if stderr_lines else f"Command failed: {e.cmd[0]}"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


# ── Routes ──

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/auth")
def auth_page():
    return render_template("auth.html")


@app.route("/process", methods=["POST"])
@optional_auth
def process():
    data = request.get_json()
    youtube_url = data.get("url", "").strip()

    if not youtube_url or ("youtube.com" not in youtube_url and "youtu.be" not in youtube_url):
        return jsonify({"error": "Please provide a valid YouTube URL"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "stage": "Starting...",
                    "error": None, "progress": None}

    thread = threading.Thread(
        target=process_job,
        args=(job_id, youtube_url),
        kwargs={"user_id": g.user_id},
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/upload", methods=["POST"])
@optional_auth
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported format. Please upload: "
                        f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    job_id = str(uuid.uuid4())
    work_dir = TMP_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    saved_path = work_dir / f"input{ext}"
    file.save(str(saved_path))

    jobs[job_id] = {"status": "processing", "stage": "Starting...",
                    "error": None, "progress": None}

    thread = threading.Thread(
        target=process_job,
        args=(job_id,),
        kwargs={"source_label": file.filename, "user_id": g.user_id},
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/library")
def library_page():
    """Render the library page (auth check happens client-side)."""
    return render_template("library.html")


@app.route("/api/library")
@require_auth
def api_library():
    """Return the authenticated user's completed extractions (newest first)."""
    result = (
        sb.table("extractions")
        .select("id, created_at, source_type, source_label, piano_path, orchestra_path")
        .eq("status", "done")
        .eq("user_id", g.user_id)
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )

    items = []
    for row in result.data:
        items.append({
            "id": row["id"],
            "created_at": row["created_at"],
            "source_type": row["source_type"],
            "source_label": row["source_label"],
            "piano_url": sb.storage.from_(STORAGE_BUCKET).get_public_url(row["piano_path"]),
            "orchestra_url": sb.storage.from_(STORAGE_BUCKET).get_public_url(row["orchestra_path"]),
        })

    return jsonify(items)


@app.route("/api/library/<extraction_id>", methods=["PATCH"])
@require_auth
def api_rename_extraction(extraction_id: str):
    """Rename an extraction's source_label. Only the owner can rename."""
    data = request.get_json()
    new_label = (data.get("source_label") or "").strip()
    if not new_label:
        return jsonify({"error": "source_label is required"}), 400

    # Update only if the user owns this extraction
    result = (
        sb.table("extractions")
        .update({"source_label": new_label})
        .eq("id", extraction_id)
        .eq("user_id", g.user_id)
        .execute()
    )

    if not result.data:
        return jsonify({"error": "Not found or not authorized"}), 404

    return jsonify({"ok": True, "source_label": new_label})


if __name__ == "__main__":
    app.run(debug=True)
