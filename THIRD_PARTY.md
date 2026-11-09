# Third-party sources and release scope

The root MIT license covers original project source only. It does not replace
the licenses of vendor IP, third-party libraries, models, recovered screenshots
or research samples.

| Component | Source | Treatment |
|---|---|---|
| MediaPipe Tasks Vision 1.1.0 | https://github.com/google-ai-edge/mediapipe | Package metadata declares Apache-2.0; retain upstream notices. Browser WASM and model binaries are not committed here. |
| Hand Landmarker model | https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker | Downloaded model asset; redistribution terms must be verified separately. |
| OpenCV | https://github.com/opencv/opencv | PS image processing and optional DNN runtime; installed versions differ on PC and board. |
| OpenCV Zoo MediaPipe palm and handpose models | https://github.com/opencv/opencv_zoo | Apache-2.0 model-directory license; large model binaries are excluded from this repository. |
| NumPy | https://github.com/numpy/numpy | Numerical processing; upstream license applies. |
| AMD FFT and Zynq IP | https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vivado.html | Vendor IP is subject to AMD terms. Reproducible source and Tcl are shared instead of a Vivado installation. |
| Optional Ultralytics export tools | https://github.com/ultralytics/ultralytics | Upstream software/model licensing must be checked for the chosen version and weights. No pretrained YOLO fretboard weights are included. |
| Albert guitar research samples | https://github.com/LincLabUCCS/Albert | Research reference only; no release/training rights established. Not included. |
| Chord recognition research | https://github.com/AlbertMitjans/chord-detection | Research reference; no checkpoint or copied implementation is included. |

The chord and scale diagrams are generated from musical intervals and
separately written code. The optional YOLO interface contains no trained
fretboard weights. Generic COCO labels must not be presented as trained
fretboard detection.

Personal recordings, complete datasets, competition screenshots containing
registration details and diagnostics with local user paths are intentionally
excluded. The repository is a curated engineering and evidence package, not a
mirror of the private working directory.
