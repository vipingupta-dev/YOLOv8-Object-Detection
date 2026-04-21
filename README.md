# YOLOv8 Object Detection
A Python project that detects 80+ objects in images, batch image folders, and videos using a pretrained YOLOv8 model.
What makes it different from a simple detection script is that it runs a full structured pipeline — input validation, inference, post-processing, visualization, batch processing, and video detection — all wired into a clean Gradio web UI where you can upload files, adjust settings, and download results. Built in Google Colab using YOLOv8 pretrained on the COCO dataset. No custom training required.

---

## What it does
You give it an image, a folder of images, or a video file. The pipeline validates your input, runs YOLOv8 inference, draws bounding boxes with class labels and confidence scores, summarizes all detections, and saves annotated outputs. It also lets you filter by class, adjust confidence threshold, and switch between fast and accurate model variants — all from a web UI.

---

## How it works
The pipeline runs through these steps:

1. **Setup** — installs dependencies, checks GPU/CPU, creates project folder structure
2. **Config** — all settings in one centralized block, nothing hardcoded anywhere else
3. **Load model** — downloads YOLOv8 pretrained weights, runs sanity check on a blank image
4. **Input handling** — accepts file path or URL, validates format, auto-resizes large images, handles all bad inputs gracefully
5. **Inference** — runs YOLOv8 predict, records inference time, extracts boxes + labels + confidence scores
6. **Post-processing** — structures raw detections into clean dicts, sorts by confidence, handles zero-detection case
7. **Visualization** — draws boxes and labels on image with unique color per class, side-by-side comparison in notebook
8. **Summary** — prints class breakdown table, top-5 detections, avg confidence, inference time
9. **Save outputs** — saves annotated image, JSON results, and CSV summary with auto-timestamped filenames
10. **Batch processing** — processes entire image folder with progress bar, saves per-image results + combined CSV
11. **Video detection** — processes video frame by frame with frame skipping, saves annotated output video
12. **UI** — launches a Gradio web app with Image Detection and Video Detection tabs

---

## Test Run Output

```
[TEST] Running pipeline on sample image...
[IMAGE] bus.jpg | 810x1080 | 2562.9 KB
[INFER] 335.4ms | 4 detections
  bus                   87.3%  bbox=[22, 231, 805, 756]
  person                86.6%  bbox=[48, 398, 245, 902]
  person                85.3%  bbox=[669, 392, 809, 877]
  person                82.5%  bbox=[221, 405, 344, 857]
[POST] 4 object(s) found across 2 class(es).
[SAVED]
  image  → outputs/bus_20260327_143327.jpg
  json   → outputs/bus_20260327_143327.json
  csv    → outputs/bus_20260327_143327_summary.csv
[TEST PASSED] 4 objects | 335.4ms
```

---

## Detection Capabilities

| Input Type | Supported Formats | Output |
|---|---|---|
| Single image | .jpg, .jpeg, .png, .bmp, .webp | Annotated image + JSON + CSV |
| Image URL | Any direct image URL | Annotated image + JSON + CSV |
| Batch folder | Mixed image files | Per-image results + combined CSV |
| Video file | .mp4, .avi, .mov | Annotated output video |

Model detects **80 COCO classes** including: person, car, bus, truck, bicycle, motorcycle, dog, cat, bottle, chair, and 70 more.

---

## Model Options

| Model | Speed | Accuracy | Best For |
|---|---|---|---|
| yolov8n | Fast (~335ms on CPU) | Good | Colab free tier, quick testing |
| yolov8m | Slower | Better | When accuracy matters more |

---

## Tech stack
- Python
- ultralytics (YOLOv8)
- OpenCV
- Pillow
- numpy, pandas
- matplotlib
- tqdm
- Gradio
- Google Colab

---

## How to run it

**Option 1 — Google Colab (easiest)**
- Open `object_detection.py`
- Upload it to colab.research.google.com as a notebook or paste cells manually
- Uncomment and run the `!pip install` line at the top first, then restart runtime
- Run all remaining cells top to bottom
- The last cell launches a Gradio web app with a public shareable link

**Option 2 — Run locally**
Make sure you have Python installed then run:
```
pip install -r requirements.txt
```
Then run the script:
```
python object_detection.py
```

---

## Requirements
All dependencies are listed in `requirements.txt`. Main ones are:
```
ultralytics
opencv-python
Pillow
numpy
pandas
matplotlib
tqdm
gradio
requests
torch
```

---

## Project structure
```
yolov8-object-detection/
│
├── object_detection.py   # main script — all steps 0-12
├── requirements.txt      # all dependencies
├── README.md             # this file
└── .gitignore            # files excluded from git
```

---

## What I learned building this

The biggest lesson was around input validation — real-world images come in wildly different formats, sizes, and sources (file paths, URLs, numpy arrays, PIL objects), and a pipeline that doesn't handle all of them will break constantly in a UI. Building one clean `validate_and_load()` function that handles every input type and fails gracefully made the whole rest of the pipeline much more stable.

I also learned that frame skipping in video detection is not just an optimization — it is necessary on Colab's free CPU tier. Processing every frame of even a short video at 335ms per frame makes the output unwatchable. Skipping every other frame and carrying the last annotated frame forward keeps the output smooth while cutting processing time in half.

---

## Author
Vipin Gupta
vipingupta.dev@gmail.com
github.com/vipingupta-dev
