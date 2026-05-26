# Face Blur Pipeline

This project is my implementation for the assignment:

- Input: short surveillance-style videos (10-15 sec) or static images
- Challenge: compression artifacts, poor lighting, motion blur
- Goal: detect and blur all visible human faces

I built a working local web app (Flask) that takes an image or video and returns a blurred output.

## What I Built

- Local web UI for upload and output preview/download
- Face detection using RetinaFace (ResNet50)
- Face blur on all detected faces
- Video temporal consistency to reduce flicker and improve uncertain detections
- Progress bar during video processing

## Why This Pipeline

For surveillance input, face quality can be poor. A simple frontal-face detector often misses side or partial faces. I chose RetinaFace because it is stronger for:

- side-profile faces
- partial/occluded faces
- multiple faces in one frame
- low-quality images

## Tools / Models / Frameworks Used

- Python
- Flask (backend + local web routes)
- OpenCV (read/write image/video, enhancement, blur, encoding)
- FaceXLib RetinaFace (PyTorch backend)
- HTML/CSS/JS (simple frontend)
- ffmpeg (optional but recommended for browser-compatible MP4 playback)

## End-to-End Pipeline

### 1) Input handling

- User uploads image (`.jpg/.jpeg/.png`) or video (`.mp4/.mov/.avi`)
- Backend saves file in `uploads/`

### 2) Optional enhancement (for bad quality footage)

- Denoise + CLAHE contrast normalization
- Helps with poor lighting and noisy frames

### 3) Face detection

- RetinaFace runs on image or each video frame
- Returns face boxes with confidence score

### 4) Detection filtering and merge

- Confidence threshold used for candidate selection
- NMS (Non-Max Suppression) removes duplicate overlapping boxes

### 5) Uncertain detection handling

#### Image

- If enhancement is enabled, detector runs on original + enhanced image
- Results are merged so weak but valid faces have better chance to survive

#### Video

- Confidence bands:
  - High confidence: blur immediately
  - Mid confidence: blur only if consistent across consecutive frames
- IoU-based temporal matching:
  - Tracks recent faces across frames
  - Promotes mid-confidence detections when they persist
- Grace window:
  - Keeps blur for a few frames if detection drops briefly
  - Reduces blur on/off flicker due to motion blur or compression

### 6) Blur application

- Adaptive Gaussian blur per face region
- Blur strength scales with face size

### 7) Output

- Saves processed file in `outputs/`
- UI shows preview + download link

## How Assignment Requirements Are Covered

- **Compression artifacts / poor lighting / motion blur**
  - Optional enhancement + temporal smoothing for video

- **Partial or side-profile faces**
  - RetinaFace model chosen for stronger profile/partial detection

- **Noisy / low-resolution input**
  - Enhancement + recall-oriented thresholding

- **Multiple faces in same frame**
  - Detector returns multiple boxes; each face box is blurred

- **Uncertain detections**
  - Mid-confidence detections require temporal consistency in video
  - Grace window avoids sudden missed blur during short drops

## Project Structure

- `app.py` - Flask app, async jobs, progress/status APIs
- `face_detector.py` - RetinaFace wrapper and device selection
- `pipeline.py` - image/video processing orchestration
- `temporal_filter.py` - IoU-based temporal consistency
- `blur_utils.py` - enhancement + blur utilities
- `templates/index.html` - web page
- `static/app.js` - upload, polling, progress bar, output rendering
- `static/style.css` - styling

## Setup (Linux/macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional (recommended for browser video preview):

```bash
sudo apt install ffmpeg
```

## Setup (Windows)

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install ffmpeg (recommended for browser video preview):

```powershell
winget install Gyan.FFmpeg
```

If venv activation is blocked in PowerShell, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## Run

```bash
python3 app.py
```

Windows:

```powershell
python app.py
```

Open:

`http://127.0.0.1:5000`

## How to Use

1. Upload image or video
2. Keep confidence around `0.20` for high recall
3. Enable enhancement for dark/noisy footage
4. Click `Process`
5. Watch progress bar (video), then preview and download

If a processed video plays in VLC but not in browser, install `ffmpeg` and reprocess the video.

## Device Behavior

- `auto`: uses GPU if available, else CPU
- `cpu`: force CPU
- `cuda`: use GPU if available, otherwise fallback to CPU

GPU improves speed; detection quality mostly depends on model and thresholds.
