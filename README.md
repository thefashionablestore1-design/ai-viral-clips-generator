# ClipForge AI

Turn videos you own or are authorized to edit into short-form clips. ClipForge analyzes speech with Whisper, ranks real transcript moments, then renders vertical MP4s with animated ASS captions, keyword highlights, subtle motion, and thumbnails using FFmpeg.

## Requirements

- Python 3.10+
- FFmpeg and ffprobe on `PATH`
- For URL input: a publicly accessible YouTube video that you own or are authorized to process. This app does not bypass login, private-video, DRM, age, or regional restrictions. YouTube access and reuse are subject to YouTube's terms and the video's rights.
- Faster-Whisper downloads its selected model on first use (default `base`). Set `WHISPER_MODEL` to `small`, `medium`, etc. for accuracy/speed tradeoffs.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
# Install FFmpeg separately, then:
uvicorn app:app --reload
```

Open http://localhost:8000. Upload an authorized video or submit a YouTube URL. Select 5–10 clips and wait for the background job to finish. Generated MP4s and a ZIP are downloadable from the results panel. Processing is CPU/GPU and disk intensive; keep the server running until the job completes.

For deployment, use persistent writable storage and a worker-friendly host. Set `CLIPFORGE_DATA_DIR` to a private, writable directory. Do not expose this development server directly to the public internet without authentication, rate limits, and storage cleanup.

## What the pipeline does

1. Validates media and extracts metadata with ffprobe (or downloads only ordinary, publicly accessible YouTube media with yt-dlp; no browser cookies or auth workarounds).
2. Transcribes the whole audio track with Faster-Whisper. If speech transcription fails or there is no speech, the job fails rather than inventing moments.
3. Scores real speech windows for energetic language, complete thoughts, and a strong opening; selects non-overlapping moments from the transcript.
4. Uses FFmpeg to create 9:16 center-cropped MP4s, trim detected silence, burn animated word/phrase captions with highlighted terms, and write a representative thumbnail. Captions, titles, descriptions, and hashtags derive from the actual transcript.

The 9:16 crop is a centered crop, not AI subject tracking. Review each export before publishing. Clip selection and “viral” scores are heuristics, not a guarantee of performance. The tool does not remove copyright restrictions or grant permission to reuse a video.

## API

- `POST /api/jobs` multipart `file`, or JSON `{ "url": "https://www.youtube.com/watch?v=...", "clip_count": 5 }`
- `GET /api/jobs/{job_id}` job state and completed clip metadata
- `GET /api/clips/{job_id}/{filename}` clip/thumbnail asset
- `GET /api/jobs/{job_id}/download-all` ZIP of completed MP4s

Set `CLIPFORGE_MAX_UPLOAD_MB` (default 1024) and `WHISPER_MODEL` to tune local operation. Temporary jobs are stored under `CLIPFORGE_DATA_DIR` (default `./data`).
