"""Fixture preparation only; requires the producer checkout and its environment."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from simcompare.adapters.numeric import load_dataset
from simcompare.electrode_map import compute_electrode_maps
from simcompare.roi_export import export_review_rois

repo = Path('/Users/jonghyeonlee/git/particle-sim-compare')
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == 'f5e86006d72faf16c3bb747051325fbba4b0f9fd'
root = repo / 'tests/fixtures/review_roi_v1'
out = Path(tempfile.mkdtemp(prefix='psdl2-producer-'))
ds = load_dataset(root)
common = {'product_id': 'PRODUCT', 'design_version': 'DESIGN', 'frame_id': 'FRAME'}
export_review_rois(ds, root, out, maps=compute_electrode_maps(ds, min_count=3),
    selections={'map_region_ids': 'all', 'manual': [
        {**common, 'case_id': 'A', 'center_xy_um': [102.5, 201.5]},
        {**common, 'center_xy_um': [102.5, 201.5]}]},
    asset_manifest={'path': 'review_assets.json', 'sha256': hashlib.sha256((root / 'review_assets.json').read_bytes()).hexdigest()})
target = Path(__file__).resolve().parent
for p in out.iterdir():
    if p.is_file():
        shutil.copy2(p, target / p.name)
payload = json.loads((out / 'review_rois.json').read_text())
for row in payload['rois']:
    assets = row.get('source_assets')
    if not assets:
        continue
    refs = [v['asset'] for v in assets['layouts']] + [assets['cnt'], assets['valid']] + assets['via']
    for ref in refs:
        dest = target / ref['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / ref['path'], dest)
print(f'Exported unchanged producer output from {out}')
