"""Create the software-only board update and its integrity manifest."""
import hashlib
from io import BytesIO
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
files = {name: root/'pynq'/name for name in
         ('hand_tracking.py', 'palm_graph.py', 'vision_pl.py', 'vision_pc_bridge.py',
          'hand_region.py', 'roi_fretboard.py', 'auto_fretboard.py', 'neck_region.py')}
files.update({str(Path('palm_execution')/name).replace('\\', '/'): root/'models/hand/palm_execution'/name
              for name in ('graph.json', 'weights.npz')})
files['deploy.sh'] = root/'scripts/deploy_realtime_update.sh'
files['native/vision_transport.c'] = root/'pynq/native/vision_transport.c'
files['benchmark_palm_graph_board.py'] = root/'diagnostics/benchmark_palm_graph_board.py'
manifest = ''.join(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+name+'\n'
                   for name, path in files.items()).encode('ascii')
with tarfile.open(root/'diagnostics/realtime_update.tar.gz', 'w:gz') as archive:
    for name, path in files.items():
        archive.add(path, arcname=name)
    item = tarfile.TarInfo('SHA256SUMS')
    item.size = len(manifest)
    archive.addfile(item, BytesIO(manifest))
print('diagnostics/realtime_update.tar.gz')
