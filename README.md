# PPE Monitor — Helmet & Reflective Jacket Detection

This repository implements a real-time Personal Protective Equipment (PPE) monitoring system that detects people and associates PPE items (helmets and reflective vests/jackets) to persons in video streams. It combines YOLO-style object detection (Ultralytics/ONNX) with a lightweight IoU tracker and heuristic association rules to produce stable per-person PPE presence results and optional email alerts.

## Key Goals

- Provide a practical, deployable monitor for helmet and reflective vest compliance on camera streams.
- Isolate and document the theoretical decisions (detection thresholds, IoU tracking, association gates, temporal smoothing).
- Offer evaluation tooling to measure detector precision / recall on YOLO-format datasets.

## Repository Layout

- `main.py` — Core application: detectors, tracker, association, visualization, CLI.
- `best.pt`, `yolo12n.pt` — Example model weights present in the repo root (user-provided or exported weights).
- `dataset/` — YOLO-format dataset: `images/` and `labels/` under `train/`, `valid/`, `test/` subfolders.
- `output/` — Intended directory for saved outputs (created by runtime options).
- `test_videos/` — Example video inputs for testing the stream mode.
- `requirements.txt` — Python dependencies.

See the `dataset/safety-Helmet-Reflective-Jacket` folder for a concrete label layout (YOLO `.txt` with class 0=`helmet`, 1=`vest`).

## Design Overview — Theory and Rationale

1) Detection Backends

- Ultralytics YOLO: When available, the code uses Ultralytics `YOLO` model wrappers for both person detection and PPE detection. These provide a modern, convenient inference API and access to `.pt` weights.
- ONNX fallback: A simple OpenCV DNN-based ONNX parser is provided (`PPEOnnx`) for cases where a model is exported to ONNX. This allows faster deployments on platforms without the Ultralytics package.

Why two backends? Ultralytics provides convenience and advanced postprocessing; ONNX provides portability and lightweight runtime options.

2) Person vs PPE Detection

- The system separates person detection from PPE detection. Person bounding boxes define regions (head and torso) used to gate PPE assignment. This separation increases robustness when PPE detections are noisy or when people are occluded.

3) Association Heuristics

- Greedy one-to-one matching assigns detected helmets to `head` regions and vests to `torso` regions.
- Matching score mixes regional IoU and the fraction of PPE inside the person box. Association requires meeting minimum IoU gates (configurable via CLI) and a minimum inside-person fraction.

Rationale: pure spatial proximity alone often mis-associates overlapping people; gating by semantic region (head/torso) and inside-fraction reduces false positives.

4) Tracking and Temporal Smoothing

- Tracker: A simple IoU-based greedy tracker (`IoUTracker`) maintains persistent IDs for detected people across frames. The tracker uses an IoU threshold and a max-age (frames) parameter.
- Smoother: `PPESmoother` maintains short per-track histories and applies a majority rule over a temporal window to stabilize noisy per-frame PPE presence.

Why this approach? Lightweight trackers are generally sufficient when identity permanence is needed and when running on resource-constrained hardware. Temporal smoothing reduces alert flapping from intermittent detection failures.

5) Alerts

- Optional email alerts attach a cropped image and send notifications for tracks missing helmets. SMTP settings and cooldowns are configurable via CLI flags.

## Usage

Run the monitor against a video source (RTSP stream, local camera index or file). Example:

```
python main.py --source 0 --ppe-weights best.pt --person-weights yolo12n.pt --conf-helmet 0.55 --conf-vest 0.60
```

Key CLI options (see `main.py` for full list):

- `--source` — video source (RTSP URL, file, or camera index)
- `--ppe-weights` — Ultralytics `.pt` weights for PPE model (helmet, vest)
- `--ppe-onnx` — OR provide an ONNX model path and `--class-names`
- `--person-weights` — Ultralytics person detector weights (default: `yolo12n.pt`)
- `--conf-helmet`, `--conf-vest` — per-class confidence thresholds to reduce false detections
- `--nms-iou` — NMS IoU used in PPE backend
- `--track-iou`, `--track-max-age` — tracker tuning params
- `--save-vis` — directory to save annotated frames
- `--eval-root` — run evaluation on a YOLO-format folder (e.g., `dataset/.../valid`) and exit

Evaluation example (compute per-class P/R on a labeled set):

```
python main.py --eval-root dataset/safety-Helmet-Reflective-Jacket/valid --ppe-weights best.pt
```

The evaluation routine compares predicted PPE boxes to ground-truth boxes (IoU ≥ 0.5) and logs precision/recall suggestions.

## Implementation Notes

- Head/torso regions: Person bounding boxes are split into heuristic head and torso rectangles (26% for head height, 52% for torso). These were chosen empirically for typical person aspect ratios but can be tuned.
- Association thresholds: Configurable gates control the minimal IoU overlap and minimal inside-person fraction required for assignment; raising thresholds reduces false positives but may lower recall.
- Filtering: Small or extreme-aspect boxes are filtered out early to drop implausible detections.

## Dependencies

Install using:

```
pip install -r requirements.txt
```

Typical packages used: `opencv-python`, `torch`, `ultralytics` (optional), `numpy` and `onnxruntime` if using ONNX.

## Extending / Training

- Training: This repo expects YOLO-format data (images + text labels). You can fine-tune YOLO (Ultralytics or other codebases) on `dataset/...` with class mapping `0=helmet, 1=vest`.
- Export: For lightweight runtime, export trained weights to ONNX and use `--ppe-onnx` with `--class-names`.
- Suggested workflow: train on combined diverse scenes, validate on the `valid/` split, and tune `--conf-helmet`/`--conf-vest` using the `--eval-root` output.

## Practical Deployment Tips

- Resize inputs and increase `--skip-frames` to reduce CPU/GPU usage on edge devices.
- Prefer ONNX + `onnxruntime` for CPU-only deployments; keep Ultralytics for GPU servers.
- If many false positives occur near frame edges, enable stricter `--min-box-area` or `--track-edge-cleanup`.

## Citations and Background Reading

- YOLO family: Redmon, et al.; single-shot detectors and subsequent YOLO improvements.
- Ultralytics YOLO: Ultralytics repository and docs for model management and training.

Please cite the original model repositories and papers if using outputs of this project in publications.

## License

This project inherits the license of included third-party models and libraries. Add or update a `LICENSE` file to declare your preferred project license.

---

If you want, I can also:

- Add a short `README` example screenshot and sample `docker`/`conda` setup.
- Create a minimal `Dockerfile` or `venv` instructions for deployment.

Created by the PPE Monitor tooling — see `main.py` for implementation details.
