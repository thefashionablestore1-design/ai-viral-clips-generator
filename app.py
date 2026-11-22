import asyncio
import json
import os
import re
import shutil
import subprocess
import threading
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

ROOT = Path(__file__).parent
DATA = Path(os.getenv("CLIPFORGE_DATA_DIR", ROOT / "data")).resolve()
DATA.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD = int(os.getenv("CLIPFORGE_MAX_UPLOAD_MB", "1024")) * 1024 * 1024
MODEL = os.getenv("WHISPER_MODEL", "base")
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
app = FastAPI(title="ClipForge AI", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class URLJob(BaseModel):
    url: HttpUrl
    clip_count: int = 5


def run(cmd: list[str], timeout: int = 3600) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError((proc.stderr or "Command failed")[-2500:])
    return proc.stdout


def update(job_id: str, **fields):
    with LOCK:
        JOBS[job_id].update(fields)


def check_count(count: int):
    if not 5 <= count <= 10:
        raise HTTPException(400, "Choose between 5 and 10 clips.")


def safe_video_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        return parsed.scheme == "https" and (host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com"))
    except Exception:
        return False


def extract_video(url: str, destination: Path):
    # Deliberately do not use cookies, credentials, proxy rotation, or restriction bypasses.
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("URL processing requires yt-dlp. Install requirements.txt.") from exc
    options = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "merge_output_format": "mp4",
        "outtmpl": str(destination / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 20,
        "max_filesize": MAX_UPLOAD,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
        if not path.exists():
            candidates = list(destination.glob("source.*"))
            if not candidates:
                raise RuntimeError("The video could not be downloaded.")
            path = candidates[0]
        return path, info.get("title", "Video")


def media_info(path: Path) -> dict:
    raw = run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height", "-of", "json", str(path)])
    info = json.loads(raw)
    duration = float(info.get("format", {}).get("duration", 0))
    if duration < 15:
        raise RuntimeError("The source video must be at least 15 seconds long.")
    if not any(s.get("codec_type") == "video" for s in info.get("streams", [])):
        raise RuntimeError("No video stream was found.")
    return {"duration": duration, "streams": info.get("streams", [])}


def transcribe(path: Path):
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(MODEL, device="auto", compute_type="int8")
        segments, _ = model.transcribe(str(path), word_timestamps=True, vad_filter=True)
        return [s for s in segments]
    except Exception as exc:
        raise RuntimeError(f"Speech transcription failed: {exc}") from exc


HOOKS = {"secret", "mistake", "never", "finally", "insane", "crazy", "best", "worst", "truth", "why", "how", "imagine", "actually", "instead", "impossible", "wait", "watch", "because", "realized", "changed"}

def choose_clips(segments, count: int, duration: float):
    rows = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            rows.append({"start": float(seg.start), "end": float(seg.end), "text": text, "words": getattr(seg, "words", None)})
    if not rows:
        raise RuntimeError("No speech was detected; no clips were fabricated.")
    # Grow transcript-centered windows to 15–60 seconds; score only actual spoken content.
    candidates = []
    for i, row in enumerate(rows):
        start = row["start"]
        end = min(start + 35, duration)
        group = [r for r in rows[i:] if r["start"] < end and r["end"] > start]
        if not group:
            continue
        actual_start, actual_end = max(0, group[0]["start"]), min(duration, group[-1]["end"])
        if actual_end - actual_start < 15:
            group = [r for r in rows if r["start"] >= actual_start and r["start"] < actual_start + 60]
            if not group:
                continue
            actual_end = min(duration, group[-1]["end"])
        actual_end = min(actual_end, actual_start + 60)
        if actual_end - actual_start < 15:
            continue
        text = " ".join(r["text"] for r in group)
        words = re.findall(r"[\w']+", text.lower())
        if len(words) < 12:
            continue
        hook = re.findall(r"[\w']+", group[0]["text"].lower())
        score = 1.0 + sum(w in HOOKS for w in words) * 0.6 + sum(w in HOOKS for w in hook) * 1.8
        score += min(len(words) / max(actual_end - actual_start, 1), 4) * .5
        score += sum(c in text for c in "?!") * .2
        score += .5 if text.rstrip().endswith((".", "?", "!")) else 0
        candidates.append({"start": actual_start, "end": actual_end, "text": text, "score": score, "rows": group})
    candidates.sort(key=lambda c: c["score"], reverse=True)
    selected = []
    for candidate in candidates:
        if all(candidate["end"] <= c["start"] - 3 or candidate["start"] >= c["end"] + 3 for c in selected):
            selected.append(candidate)
            if len(selected) == count:
                break
    if not selected:
        raise RuntimeError("Not enough distinct speech moments between 15 and 60 seconds were found.")
    selected.sort(key=lambda c: c["start"])
    return selected


def ass_time(seconds: float) -> str:
    cs = int(max(0, seconds) * 100)
    return f"{cs//360000}:{(cs//6000)%60:02}:{(cs//100)%60:02}.{cs%100:02}"


def escape_ass(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", " ")


def make_ass(clip: dict, path: Path):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,66,&H00FFFFFF,&H0000FFFF,&H00101010,&H90000000,-1,0,0,0,100,100,1,0,1,5,2,2,70,70,350,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for row in clip["rows"]:
        start = max(0, row["start"] - clip["start"])
        end = max(start + .3, min(clip["end"] - clip["start"], row["end"] - clip["start"]))
        text = escape_ass(row["text"].strip())
        terms = re.findall(r"\b(?:never|secret|mistake|insane|best|worst|truth|impossible|actually|crazy|why|finally)\b", text, re.I)
        if terms:
            term = terms[0]
            text = re.sub(rf"\b({re.escape(term)})\b", r"{{\\c&H0000FFFF&}}\1{{\\c&H00FFFFFF&}}", text, count=1, flags=re.I)
        events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{{\\fad(100,100)}}{text}")
    path.write_text(header + "\n".join(events), encoding="utf-8")


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-")[:48] or "clip"


def render_clip(source: Path, folder: Path, clip: dict, index: int, title: str):
    start, end = clip["start"], clip["end"]
    output = folder / f"clip-{index:02}.mp4"
    ass = folder / f"clip-{index:02}.ass"
    thumb = folder / f"clip-{index:02}.jpg"
    make_ass(clip, ass)
    # Scale to fill portrait canvas and center-crop; slight animated push-in adds polish.
    subtitle_path = str(ass.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    video_filter = f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,zoompan=z='min(zoom+0.0004,1.035)':d=1:s=1080x1920:fps=30,ass='{subtitle_path}'"
    run(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end-start:.3f}", "-vf", video_filter, "-af", "silenceremove=start_periods=1:start_duration=0.15:start_threshold=-48dB:detection= peak,areverse,silenceremove=start_periods=1:start_duration=0.15:start_threshold=-48dB,areverse,loudnorm=I=-16:TP=-1.5:LRA=11", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(output)], timeout=900)
    run(["ffmpeg", "-y", "-ss", f"{max(start, (start+end)/2):.3f}", "-i", str(source), "-frames:v", "1", "-vf", "scale=540:960:force_original_aspect_ratio=increase,crop=540:960", "-q:v", "3", str(thumb)])
    words = re.findall(r"[\w']+", clip["text"])
    title_text = " ".join(words[:8]).strip(".,!? ") or title
    tags = list(dict.fromkeys([w.lower() for w in words if len(w) > 5 and w.lower() not in {"because", "really", "something", "people", "think"}]))[:3]
    return {"filename": output.name, "thumbnail": thumb.name, "title": title_text[:70], "description": f"A standout moment from {title}.\n\n#shorts #viral " + " ".join("#" + re.sub(r"[^\w]", "", tag) for tag in tags), "start": round(start, 1), "duration": round(end-start, 1), "score": round(clip["score"], 1), "transcript": clip["text"]}


def process_job(job_id: str, source: Path, title: str, count: int):
    folder = DATA / job_id
    try:
        update(job_id, status="analyzing", progress=8, message="Checking video and audio…")
        info = media_info(source)
        update(job_id, progress=18, message="Transcribing the full video with Whisper…")
        segments = transcribe(source)
        update(job_id, progress=48, message="Finding distinct moments from the transcript…")
        clips = choose_clips(segments, count, info["duration"])
        result = []
        for i, clip in enumerate(clips, 1):
            update(job_id, progress=48 + int(48 * (i-1) / len(clips)), message=f"Rendering clip {i} of {len(clips)}…")
            result.append(render_clip(source, folder, clip, i, title))
        zippath = folder / "clips.zip"
        with zipfile.ZipFile(zippath, "w", zipfile.ZIP_DEFLATED) as zf:
            for clip in result:
                zf.write(folder / clip["filename"], clip["filename"])
        update(job_id, status="complete", progress=100, message=f"Ready — {len(result)} real moments selected.", title=title, clips=result, source_duration=round(info["duration"], 1))
    except Exception as exc:
        update(job_id, status="error", progress=0, message=str(exc)[:800])
    finally:
        try:
            source.unlink(missing_ok=True)
        except Exception:
            pass


def launch(source: Path, title: str, count: int) -> str:
    job_id = uuid.uuid4().hex
    folder = DATA / job_id
    folder.mkdir(parents=True, exist_ok=True)
    JOBS[job_id] = {"id": job_id, "status": "queued", "progress": 2, "message": "Job queued…", "clips": [], "title": title}
    threading.Thread(target=process_job, args=(job_id, source, title, count), daemon=True).start()
    return job_id


@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/jobs")
async def create_upload_job(background_tasks: BackgroundTasks, file: UploadFile = File(...), clip_count: int = 5):
    check_count(clip_count)
    ext = Path(file.filename or "video").suffix.lower()
    if ext not in {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}:
        raise HTTPException(400, "Upload a supported video file: MP4, MOV, MKV, WebM, M4V, or AVI.")
    temp = DATA / f"upload-{uuid.uuid4().hex}{ext}"
    size = 0
    try:
        with temp.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, f"Maximum upload size is {MAX_UPLOAD // (1024*1024)} MB.")
                out.write(chunk)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    job_id = launch(temp, Path(file.filename).stem or "Uploaded video", clip_count)
    return {"id": job_id}


@app.post("/api/jobs/url")
def create_url_job(body: URLJob):
    check_count(body.clip_count)
    url = str(body.url)
    if not safe_video_url(url):
        raise HTTPException(400, "Only standard HTTPS YouTube video URLs are supported.")
    job_id = uuid.uuid4().hex
    folder = DATA / job_id
    folder.mkdir(parents=True, exist_ok=True)
    JOBS[job_id] = {"id": job_id, "status": "downloading", "progress": 3, "message": "Checking publicly accessible video…", "clips": []}
    def download_and_run():
        try:
            source, title = extract_video(url, folder)
            update(job_id, status="analyzing", progress=8, message="Video ready; starting analysis…", title=title)
            process_job(job_id, source, title, body.clip_count)
        except Exception as exc:
            update(job_id, status="error", progress=0, message=f"Could not access this video: {str(exc)[:650]}")
    threading.Thread(target=download_and_run, daemon=True).start()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found (jobs are in memory and may not survive a server restart).")
    return job


@app.get("/api/clips/{job_id}/{filename}")
def clip_asset(job_id: str, filename: str):
    if not re.fullmatch(r"clip-\d{2}\.(?:mp4|jpg)", filename):
        raise HTTPException(400, "Invalid asset name.")
    path = DATA / job_id / filename
    if not path.is_file():
        raise HTTPException(404, "Asset not found.")
    return FileResponse(path, filename=filename)


@app.get("/api/jobs/{job_id}/download-all")
def download_all(job_id: str):
    path = DATA / job_id / "clips.zip"
    if not path.is_file():
        raise HTTPException(404, "The clip bundle is not ready yet.")
    return FileResponse(path, filename="clipforge-clips.zip", media_type="application/zip")
