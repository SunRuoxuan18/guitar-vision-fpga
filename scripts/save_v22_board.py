"""Capture the accepted static deployment without restarting its service."""
import hashlib
import json
import pathlib
import subprocess
import tarfile


root = pathlib.Path('/home/xilinx/av_system')
release_id = 'fretlens_v22_20261023'
destination = pathlib.Path('/home/xilinx/releases') / release_id
destination.mkdir(parents=True, exist_ok=False)
expected = {
    'av_system.bit': '206564927891851853fa4e634671e7cfe1a030695afef3ee85e1e89e425d37ac',
    'av_system.hwh': 'a745f0c16a39969ed2d10baa05efa3217273d83a6fc87659b7b96e41b0ccc100',
    'hand_region.py': '15296c4788b78d869ad9fe1f49e2df7ffc3950c9664c695606912e16d3c30090',
    'fretboard_tracker.py': '4e22ea3e8e4900c712d7df8ac39c62401a737728ac2337a522023d980c032492',
    'roi_fretboard.py': 'a325bb352d88659b5688cab776f67feefd3fe3403ed60818bb108c62cc97c532',
    'vision_pc_bridge.py': '6d690d1990943aa2694fe997e502a688f6e65c20c6a648d9fd110dac6217ea04',
    'tracking_native.py': 'fc63228540e848b6cb3afeb03cacfddb83e7fe5d0967e97223f7e13e431b6335',
    'libtracking_cv.so': '9ef0d8fd2c77ebb2c17623d68e2778ead6130929cc6ea8c067b3d3602aa74085',
}
for name, digest in expected.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name

archive_path = destination / 'board_runtime.tar.gz'
with tarfile.open(str(archive_path), 'x:gz', compresslevel=1) as archive:
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part in ('previous_units', '__pycache__') for part in relative.parts):
            continue
        if not path.is_file() or path.suffix in ('.log', '.pyc'):
            continue
        archive.add(str(path), arcname='av_system/' + relative.as_posix(), recursive=False)
    for base in ('/etc/systemd/system', '/run/systemd/system'):
        for path in sorted(pathlib.Path(base).glob('fretlens-av*')):
            if path.is_file():
                archive.add(str(path), arcname=str(path).lstrip('/'), recursive=False)
            elif path.is_dir():
                archive.add(str(path), arcname=str(path).lstrip('/'))

record = {
    'release_id': release_id,
    'status': 'latest_user_accepted',
    'archive': str(archive_path),
    'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    'verified_deployment': expected,
    'scope': 'Static application, models, native libraries, PL files and service configuration; excludes OS, Python environment and live state.',
}
(destination / 'board_snapshot.json').write_text(json.dumps(record, indent=2) + '\n')
subprocess.run(['systemctl', 'is-active', 'fretlens-av.service'], check=True)
print(json.dumps(record), flush=True)
