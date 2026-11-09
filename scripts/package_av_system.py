"""Package the verified combined bitstream and its matching board services."""

import hashlib
import argparse
import json
from pathlib import Path
import tarfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',default='output/av_system')
    parser.add_argument('--archive',default='diagnostics/av_system_deploy.tar.gz')
    args = parser.parse_args()
    release = root / args.release
    hashes = {suffix: hashlib.sha256((release / ("av_system" + suffix)).read_bytes()).hexdigest()
              for suffix in (".bit", ".hwh")}
    manifest = {"design": "AVS1", "sha256": hashes,
                "vision_clock_mhz": 100, "audio_clock_mhz": 125,
                "audio_address": "0x43c20000", "flow_address": "0x43c00000",
                "dma_address": "0x40400000"}
    if 'hand_conv' in (release/'av_system.hwh').read_text(encoding='utf-8'):
        manifest['neural_address'] = '0x43c30000'
        manifest['neural_core'] = ('hand_conv_q16' if 'hand_conv_q16' in
            (release/'av_system.hwh').read_text(encoding='utf-8') else 'hand_conv')
    if 'hand_requant' in (release/'av_system.hwh').read_text(encoding='utf-8'):
        manifest['requant_address'] = '0x43c40000'
    if 'hand_dw_pw_stream' in (release/'av_system.hwh').read_text(encoding='utf-8'):
        manifest['stream_address'] = '0x43c50000'
        manifest['stream_core'] = 'hand_dw_pw_stream'
    (release / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    modules = ("av_system.py", "README_AV_SYSTEM.md", "audio_fft.py", "audio_music.py", "audio_native.py", "audio_music_server.py",
               "music_tracking.py", "test_board_music.py", "benchmark_av_audio.py",
               "vision_pc_bridge.py", "vision_pl.py", "vision_hls.py",
               "fretboard_tracker.py", "auto_fretboard.py", "roi_fretboard.py", "neck_region.py",
               "neck_model.py", "neck_color.py", "yolo_detector.py", "yolo_dfl.py",
               "hand_tracking.py", "hand_region.py", "hand_graph.py", "palm_graph.py", "hand_requant.py", "hand_stream.py", "work_priority.py", "handpose_model.py", "hand_neural_pl.py", "patch_similarity.py", "fast_jpeg.py", "visual_positions.py", "av_fusion.py", "overlay_identity.py", "benchmark_hand_neural.py")
    units = ("fretlens-av.target", "fretlens-av.service")
    archive = root / args.archive
    with tarfile.open(archive, "w:gz") as package:
        for name in ("av_system.bit", "av_system.hwh", "manifest.json"):
            package.add(release / name, arcname=name)
        for name in modules + units:
            package.add(root / "pynq" / name, arcname=name)
        package.add(root / "pynq/native/audio_fit.c", arcname="native/audio_fit.c")
        package.add(root / "pynq/native/hand_q16.c", arcname="native/hand_q16.c")
        package.add(root / "pynq/native/fast_jpeg.c", arcname="native/fast_jpeg.c")
        package.add(root / "pynq/native/vision_transport.c", arcname="native/vision_transport.c")
        package.add(root / "pynq/native/neck_region.cpp", arcname="native/neck_region.cpp")
        package.add(root / "pynq/native/neck_color.cpp", arcname="native/neck_color.cpp")
        package.add(root / "pynq/native/neck_color_ridges.cpp", arcname="native/neck_color_ridges.cpp")
        package.add(root / "pynq/native/neck_color_candidates.cpp", arcname="native/neck_color_candidates.cpp")
        package.add(root / "scripts/install_av_system.sh", arcname="install_av_system.sh")
        package.add(root / "models/hand/handpose_execution", arcname="handpose_execution")
        package.add(root / "models/hand/palm_execution", arcname="palm_execution")
        for source, target in (("palm_detection_mediapipe_2023feb_cv42.onnx", "palm.onnx"),
                               ("handpose_estimation_mediapipe_2023feb_cv42.onnx", "handpose.onnx"),
                               ("palm_detection_mediapipe_2023feb_pl.onnx", "palm_pl.onnx"),
                               ("handpose_estimation_mediapipe_2023feb_pl.onnx", "handpose_pl.onnx"),
                               ("LICENSE", "HAND_MODELS_LICENSE")):
            package.add(root / "models/hand" / source, arcname=target)
    print(json.dumps({"archive": str(archive), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
