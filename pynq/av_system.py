"""Single overlay owner for the combined AVS1 design (PYNQ 2.7)."""

import argparse
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import signal
import sys
import time

from audio_fft import AUD4_ADDRESS_RANGE, AudioFFT


AUDIO_BASE = 0x43C20000
CORES = {"vision_dma": (0x40400000, "xilinx.com:ip:axi_dma:"),
         "lk_pyramid": (0x43C00000, "xilinx.com:hls:lk_pyramid:"),
         "music": (AUDIO_BASE, "xilinx.com:module_ref:audio_music_pl:")}


def validate_layout(ip_dict, require_neural=False):
    for name, (address, kind) in CORES.items():
        description = ip_dict.get(name, {})
        if (int(description.get("phys_addr", -1)) != address
                or not str(description.get("type", "")).startswith(kind)):
            raise ValueError("Combined overlay has an invalid {} address/type".format(name))
    if int(ip_dict["music"]["addr_range"]) < AUD4_ADDRESS_RANGE:
        raise ValueError("Audio register range is too small")
    neural_names = [name for name in ('hand_conv','hand_conv_q16') if name in ip_dict]
    if require_neural or neural_names:
        if len(neural_names) != 1:
            raise ValueError('Expected exactly one hand_conv or hand_conv_q16 core')
        name = neural_names[0]
        neural = ip_dict[name]
        if (int(neural.get('phys_addr',-1)) != 0x43c30000
                or not neural.get('type','').startswith('xilinx.com:hls:'+name+':')
                or int(neural.get('addr_range',0)) < 0x100):
            raise ValueError('Invalid hand_conv address/type/range')
    if 'hand_requant' in ip_dict:
        requant = ip_dict['hand_requant']
        if (int(requant.get('phys_addr', -1)) != 0x43c40000
                or not requant.get('type', '').startswith('xilinx.com:hls:hand_requant:')
                or int(requant.get('addr_range', 0)) < 0x100):
            raise ValueError('Invalid hand_requant address/type/range')
    if 'hand_dw_pw_stream' in ip_dict:
        stream = ip_dict['hand_dw_pw_stream']
        if (int(stream.get('phys_addr', -1)) != 0x43c50000
                or not stream.get('type', '').startswith('xilinx.com:hls:hand_dw_pw_stream:')
                or int(stream.get('addr_range', 0)) < 0x100):
            raise ValueError('Invalid hand_dw_pw_stream address/type/range')


def validate_artifacts(bitfile, manifest):
    expected = json.loads(Path(manifest).read_text(encoding="utf-8"))
    actual = {}
    for suffix in (".bit", ".hwh"):
        path = Path(bitfile).with_suffix(suffix)
        actual[suffix] = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected.get("sha256", {}).get(suffix) != actual[suffix]:
            raise ValueError("Combined overlay {} SHA256 mismatch".format(suffix))
    return actual


def activate(bitfile, manifest):
    hashes = validate_artifacts(bitfile, manifest)
    from pynq import Clocks, Overlay
    overlay = Overlay(str(Path(bitfile).resolve()), download=False)
    expected = json.loads(Path(manifest).read_text(encoding='utf-8'))
    validate_layout(overlay.ip_dict,require_neural='neural_address' in expected)
    if expected.get('neural_core') and expected['neural_core'] not in overlay.ip_dict:
        raise ValueError('Manifest neural core does not match hardware handoff')
    if expected.get('requant_address') and 'hand_requant' not in overlay.ip_dict:
        raise ValueError('Manifest requant core does not match hardware handoff')
    if expected.get('stream_address') and 'hand_dw_pw_stream' not in overlay.ip_dict:
        raise ValueError('Manifest stream core does not match hardware handoff')
    overlay.download()
    # Use the running PS PLL; do not execute ps7_init or touch Linux DDR.
    Clocks.fclk0_mhz = 100
    Clocks.fclk1_mhz = 125
    clocks = [float(Clocks.fclk0_mhz), float(Clocks.fclk1_mhz)]
    if any(not math.isfinite(actual) or abs(actual-target) > .5
           for actual, target in zip(clocks, (100, 125))):
        raise RuntimeError("Combined overlay clocks are incorrect: {}".format(clocks))
    with_audio = AudioFFT(base_address=AUDIO_BASE, dev_mem=True)
    try:
        # Identity/metadata reads in AudioFFT.__init__ check the live audio core.
        # Do not freeze/read spectra here: the audio service owns snapshots.
        core_id = with_audio.mmio.read(0)
        status = with_audio.mmio.read(4)
    finally:
        with_audio.close()
    return {"design": "AVS1", "checked_at": time.time(), "sha256": hashes,
            "bitfile": str(Path(bitfile).resolve()), "clocks_mhz": clocks,
            "audio_core_id": hex(core_id), "audio_status": status,
            "addresses": {name: hex(value[0]) for name, value in CORES.items()},
            "overlay_timestamp": str(getattr(overlay.device, "timestamp", None))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitfile", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--benchmark-audio-seconds", type=float)
    parser.add_argument("--output", default="/run/fretlens-av.json")
    args = parser.parse_args()
    result = activate(args.bitfile, args.manifest)
    target = Path(args.output)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(result, separators=(",", ":")), flush=True)
    if args.self_test:
        from test_board_music import run_smoke
        driver = AudioFFT(base_address=AUDIO_BASE, dev_mem=True)
        try:
            report = run_smoke(driver)
        finally:
            driver.close()
        Path("/home/xilinx/av_system/board_smoke.json").write_text(json.dumps(report, indent=2))
        if not report["passed"]:
            raise RuntimeError("Unified audio PL smoke failed: {}".format(report.get("error")))
        print("AV_AUDIO_SELF_TEST_PASS", flush=True)
    if args.serve:
        from pynq import Device
        from overlay_identity import IdentityServer
        with IdentityServer(Device.active_device):
            supervise(args.bitfile, args.benchmark_audio_seconds)


def _audio_worker():
    from audio_music_server import main as audio_main
    audio_main(["--dev-mem", "--base-address", hex(AUDIO_BASE), "--host", "0.0.0.0",
                "--port", "8081", "--poll-interval", ".05", "--test-mode", "0", "--warmup-templates"])


def _vision_worker(bitfile):
    from vision_pc_bridge import main as vision_main
    sys.argv = ["vision_pc_bridge.py", "--bitfile", str(bitfile), "--hls-flow"]
    vision_main()


def _benchmark_audio_worker(seconds):
    from benchmark_av_audio import main as benchmark_main
    sys.argv = ["benchmark_av_audio.py", "--seconds", str(seconds),
                "--output", "/home/xilinx/av_system/audio_joint_benchmark.json"]
    benchmark_main()


def supervise(bitfile, benchmark_seconds=None):
    # This board's PYNQ Device keeps loaded-overlay identity in process memory.
    # Fork only after download completes, preserving that identity in both workers.
    # The parent remains the sole overlay owner; no worker calls download().
    context = multiprocessing.get_context("fork")
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    audio_target = _audio_worker if benchmark_seconds is None else _benchmark_audio_worker
    audio_args = () if benchmark_seconds is None else (benchmark_seconds,)
    children = [context.Process(target=audio_target, args=audio_args, name="av-audio"),
                context.Process(target=_vision_worker, args=(bitfile,), name="av-vision")]
    try:
        for child in children:
            child.start()
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        while not stop:
            for child in children:
                if child.exitcode is not None:
                    if child is children[0] and benchmark_seconds is not None and child.exitcode == 0:
                        return
                    raise RuntimeError("{} exited with code {}".format(child.name, child.exitcode))
            time.sleep(.2)
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            if child.pid is not None:
                child.join(timeout=35 if child is children[1] else 4)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=2)


if __name__ == "__main__":
    main()
