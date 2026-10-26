"""ETX real-data runners (scripts/build_etx_jobs.py): psdl2_job_db_search.py (torch) and psdl2_job_eval.py (no torch).

The synthetic world is the one of the torch smoke bundle (P0..P2 compatible, N incompatible), laid out the way a user fills the
EDIT block: relative paths below BASE_DIR, a BASE_DIR with a space and non-ASCII characters, the generated runner executed
with `-I` from another working directory. db_search runs on venv39-torch and its envelope feeds eval on venv39-notorch.
Everything is synthetic and offline; CUDA and a Linux file system are not covered.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import pytest

from test_etx_runtime import ROOT, WORKSPACE, etx_py, execute, py38, read_log, torch_py  # noqa: F401  (fixtures are used by name)

DB = ROOT / 'bundles/etx/psdl2_job_db_search.py'
EVAL = ROOT / 'bundles/etx/psdl2_job_eval.py'
DB_LOG, EVAL_LOG = 'runs/psdl2_db_search.log', 'runs/psdl2_eval.log'
PRODUCT_NAMES = ['P0', 'P1', 'P2', 'N']
DB_VALUES = {'CONFIG': 'dl2_db/config.json', 'COHORT': 'dl2_db/cohort.json', 'GRID_STEP_UM': 2.0, 'TOPK': 4, 'PER_PRODUCT_TOPK': 2,
             'DEVICE': 'cpu'}
EVAL_VALUES = {'WINDOW_UM': 6, 'NCC_TOPK': 3, 'MAX_MEMORY_MB': 8000, 'HIT_TOLERANCE_UM': 0.6, 'K': 3}


def builder():
    sys.path.insert(0, str(ROOT / 'scripts'))
    try:
        import build_etx_jobs
    finally:
        sys.path.pop(0)
    return build_etx_jobs


def smoke_world(destination):
    """The synthetic fixture of the torch smoke bundle, made by its own _prepare hook (no training, no subprocess)."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    try:
        import build_etx_smoke
    finally:
        sys.path.pop(0)
    namespace = {'hashlib': hashlib, 'json': json, 'Path': Path, 'time': time, 'sys': sys, 'DONE_SUFFIX': '.psdl2_done.json',
                 '_fail': lambda *args, **kwargs: pytest.fail(f'fixture _fail: {args}')}
    exec(compile(build_etx_smoke._hooks(), 'smoke_hooks', 'exec'), namespace)
    namespace['_prepare'](destination)
    return destination / 'psdl2_smoke'


def make_workspace(base):
    """base/dl2_db holds the inputs of db_search; the eval inputs point into the same export folders."""
    world = smoke_world(base)
    inputs = base / 'dl2_db'
    inputs.mkdir()
    shutil.copytree(world / 'templates', inputs / 'templates')
    for source, target in (('plain.json', 'db_dataset.json'), ('query.json', 'query_dataset.json'), ('lineages.json', 'lineages.json'),
                           ('assign.json', 'assign.json'), ('config.json', 'config.json'), ('cohort.json', 'cohort.json')):
        shutil.copy(world / source, inputs / target)
    (base / 'dl2_eval').mkdir()
    shutil.copy(world / 'correspondence.json', base / 'dl2_eval/correspondence.json')
    shutil.rmtree(world)
    return base


def fill(text, values):
    """Set EDIT values the way a user does: replace `NAME = FILL_ME`."""
    for name, value in values.items():
        text, count = re.subn(rf'(?m)^{name} = FILL_ME\b', lambda _: f'{name} = {value!r}', text)
        assert count == 1, name
    return text


def edit(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def db_text(base, *, values=None):
    text = DB.read_text(encoding='utf-8')
    text = fill(text, DB_VALUES if values is None else values)
    text = edit(text, 'PRODUCTS = ["productA", "productB"]', f'PRODUCTS = {PRODUCT_NAMES!r}')
    text = edit(text, 'OUT_DIR = "dl2_db/out"', 'OUT_DIR = "work/db_out"')   # a sibling folder: the patch set paths use `..`
    return text.replace('BASE_DIR = None', f'BASE_DIR = {str(base)!r}', 1)


def eval_text(base, *, values=None):
    text = EVAL.read_text(encoding='utf-8')
    text = fill(text, EVAL_VALUES if values is None else values)
    text = edit(text, 'QUERY_EXPORT = "dl2_eval/query/review_rois.json"', 'QUERY_EXPORT = "dl2_db/templates/P0/review_rois.json"')
    text = edit(text, '("productA", "dl2_eval/templates/productA/review_rois.json")', '("P1", "dl2_db/templates/P1/review_rois.json")')
    text = edit(text, '("productB", "dl2_eval/templates/productB/review_rois.json")', '("P2", "dl2_db/templates/P2/review_rois.json")')
    text = edit(text, 'SEARCH_BATCH = "dl2_db/out/search_batch.json"', 'SEARCH_BATCH = "work/db_out/search_batch.json"')
    text = edit(text, 'OUT_DIR = "dl2_eval/out"', 'OUT_DIR = "work/eval_out"')
    return text.replace('BASE_DIR = None', f'BASE_DIR = {str(base)!r}', 1)


def place(tmp_path, text, name):
    script = tmp_path / 'upload dir' / name
    script.parent.mkdir(exist_ok=True)
    script.write_text(text, encoding='utf-8')
    return script


@pytest.fixture
def workspace(tmp_path):
    return make_workspace(tmp_path / WORKSPACE)


def started(log):
    return re.findall(r' start (\w+) ', log)


# ---------------------------------------------------------------- the committed bundles

def test_bundles_are_deterministic_and_the_committed_files_are_current(tmp_path):
    first, second = tmp_path / 'a', tmp_path / 'b'
    for folder in (first, second):
        builder().main(['--out-dir', str(folder)])
    for name in ('psdl2_job_db_search.py', 'psdl2_job_eval.py'):
        assert (first / name).read_bytes() == (second / name).read_bytes() == (ROOT / 'bundles/etx' / name).read_bytes(), name
    # T9 adv:3: the unreachable "envelope does not exist yet" branch is gone (a missing envelope stops with input_missing first)
    assert 'does not exist yet' not in (first / 'psdl2_job_eval.py').read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        builder().build_db_search(first / 'psdl2_job_db_search.py')


def test_the_bundles_compile_on_python39_and_python38(etx_py, py38, tmp_path):  # noqa: F811
    program = 'import sys\nfor p in sys.argv[1:]:\n    compile(open(p, encoding="utf-8").read(), p, "exec")\nprint("compiled")\n'
    for python in (etx_py, py38):
        proc = subprocess.run([str(python), '-I', '-c', program, str(DB), str(EVAL)], capture_output=True, cwd=tmp_path)
        assert proc.returncode == 0 and b'compiled' in proc.stdout, proc.stderr


def test_edit_blocks_have_no_numeric_default_and_the_header_names_the_round():
    for path in (DB, EVAL):
        text = path.read_text(encoding='utf-8')
        edit_block = text[text.index('# ====' + '=' * 26 + ' EDIT BELOW'):text.index('EDIT ABOVE')]
        header = text[:text.index('# ====' + '=' * 26 + ' EDIT BELOW')]
        assert '변경점 ROI export' in header and '제품별 template export' in header
        assert 'review_roi_v1 manual 행 + mask_derivation.json' in header and 'PLAN D2' in header and '미확정' in header
        values = re.findall(r'(?m)^([A-Z_]+) = (\S+)', edit_block)
        # every number-like setting is FILL_ME; the only plain literals are paths, names, DEVICE and BASE_DIR/WORK_DIR/LOG_FILE
        numeric = [(n, v) for n, v in values if re.fullmatch(r'-?\d+(\.\d+)?', v)]
        assert numeric == [], numeric
        assert 'FILL_ME' in edit_block


def test_per_product_topk_none_drops_the_flag_and_a_number_keeps_it():
    def search_args(per_product):
        text = DB.read_text(encoding='utf-8')
        text = fill(text, dict(DB_VALUES, PER_PRODUCT_TOPK=per_product))
        namespace = {}
        exec(compile(text[:text.index('EDIT ABOVE')], 'edit', 'exec'), namespace)
        return dict(namespace['JOBS'])['search']
    assert '--per-product-topk' not in search_args(None)
    args = search_args(3)
    assert args[args.index('--per-product-topk') + 1] == '3'


# ---------------------------------------------------------------- the whole synthetic flow

def test_db_search_on_torch_then_eval_on_notorch_python(torch_py, etx_py, tmp_path, workspace):  # noqa: F811
    db = place(tmp_path, db_text(workspace), 'db_search.py')
    (workspace / 'dl2_db/db_dataset.grid.json').write_text('{}', encoding='utf-8')   # a stale file of another content is overwritten
    proc = execute(torch_py, db, tmp_path)   # `-I`, another working directory, BASE_DIR with a space and non-ASCII characters
    assert proc.returncode == 0, proc.err
    assert 'Traceback' not in proc.err
    log = read_log(workspace, DB_LOG)
    assert 'overwriting dl2_db/db_dataset.grid.json' in log
    for index, name in enumerate(PRODUCT_NAMES):   # adv:4: one line per product names its source export
        assert f'grid_{name}.json <- source {index}: dl2_db/templates/{name}/review_rois.json' in log
    assert 'search complete: 3 queries' in log
    assert 'python 3.9.' in log and 'torch 2.1.1' in log and 'cuda available:' in log
    assert started(log) == ['patches'] * 4 + ['make_split', 'train', 'index', 'search']
    for step in ('patches', 'make_split', 'train', 'index', 'search'):
        assert f' {step} exit 0 in ' in log
    assert 'search batch work/db_out/search_batch.json: 3 queries, 0 without candidates' in log
    assert (workspace / 'dl2_db/db_dataset.grid.json').is_file()
    manifest = json.loads((workspace / 'dl2_db/db_dataset.grid.json').read_text(encoding='utf-8'))
    assert [s['patch_sets'] for s in manifest['sources']] == [[f'../work/db_out/grid_{n}.json'] for n in PRODUCT_NAMES]
    plain = json.loads((workspace / 'dl2_db/db_dataset.json').read_text(encoding='utf-8'))
    assert all(not s.get('patch_sets') for s in plain['sources'])   # the user's manifest is not touched
    batch = json.loads((workspace / 'work/db_out/search_batch.json').read_text(encoding='utf-8'))
    assert batch['schema_version'] == 'psdl2_search_batch_v1' and batch['arguments']['per_product_topk'] == 2
    assert batch['arguments']['exclude_query_product'] is True
    for ranking in batch['rankings']:
        products = {c['product_id'] for c in ranking['candidates']}
        assert ranking['candidates'] and ranking['query_product_id'] not in products and 'N' not in products
    assert not (tmp_path / 'elsewhere/work').exists() and not (tmp_path / 'upload dir/work').exists()

    runner = place(tmp_path, eval_text(workspace), 'eval.py')
    proc = execute(etx_py, runner, tmp_path)   # no torch here
    assert proc.returncode == 0, proc.err
    assert 'Traceback' not in proc.err
    log = read_log(workspace, EVAL_LOG)
    assert 'torch unavailable' in log and 'numpy 1.23.5' in log
    assert started(log) == ['baseline_ncc', 'baseline_ncc', 'eval_position', 'eval_position']
    out = workspace / 'work/eval_out'
    for name in ('P1', 'P2'):
        assert (out / f'ncc_{name}.json').is_file() and (out / f'ncc_{name}_sheet/sheet.html').is_file()
    baseline = json.loads((out / 'eval_baseline.json').read_text(encoding='utf-8'))
    assert baseline['schema_version'] == 'psdl2_eval_position_multi_v1' and baseline['arguments']['mode'] == 'multi_baseline'
    assert baseline['arguments']['baselines'] == ['work/eval_out/ncc_P1.json', 'work/eval_out/ncc_P2.json']
    searched = json.loads((out / 'eval_search_batch.json').read_text(encoding='utf-8'))
    assert searched['arguments']['mode'] == 'search_batch' and searched['summary']['n_queries_with_truth'] == 3
    assert searched['status'] == 'available'
    assert 'work/eval_out/eval_baseline.json: status' in log and 'work/eval_out/eval_search_batch.json: status available' in log

    # a second run skips the whitelisted jobs (baseline_ncc and eval_position) and changes nothing
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    again = execute(etx_py, runner, tmp_path)
    assert again.returncode == 0, again.err
    second = read_log(workspace, EVAL_LOG).split('prepare:')[-1]
    assert second.count('skip completed') == 4 and started(second) == []
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}

    # adv:5: other bytes in the envelope (same content) -> the first three jobs are skipped, the search-batch evaluation is stale
    envelope = workspace / 'work/db_out/search_batch.json'
    envelope.write_text(json.dumps(json.loads(envelope.read_text(encoding='utf-8')), indent=4), encoding='utf-8')
    stale = execute(etx_py, runner, tmp_path)
    assert stale.returncode == 2 and 'Traceback' not in stale.err
    third = read_log(workspace, EVAL_LOG).split('prepare:')[-1]
    assert third.count('skip completed') == 3 and started(third) == []
    assert 'ERROR stale_done_marker:inputs_changed' in third and 'work/db_out/search_batch.json' in third.split('ERROR stale_done_marker')[1]


# ---------------------------------------------------------------- error paths (no job runs)

@pytest.mark.parametrize('which', ['db', 'eval'])
def test_a_remaining_fill_me_stops_before_any_job(etx_py, tmp_path, workspace, which):  # noqa: F811
    text = db_text(workspace, values={}) if which == 'db' else eval_text(workspace, values={})
    log_name = DB_LOG if which == 'db' else EVAL_LOG
    proc = execute(etx_py, place(tmp_path, text, 'runner.py'), tmp_path)
    assert proc.returncode == 2 and 'Traceback' not in proc.err
    names = ['CONFIG', 'GRID_STEP_UM', 'COHORT', 'TOPK', 'PER_PRODUCT_TOPK', 'DEVICE'] if which == 'db' else \
        ['WINDOW_UM', 'NCC_TOPK', 'MAX_MEMORY_MB', 'HIT_TOLERANCE_UM', 'K']
    log = read_log(workspace, log_name)
    for stream in (proc.err, log):
        assert f'ERROR edit_value_required: {", ".join(names)} still ' in stream
        assert 'EDIT 블록' in stream and '어떤 작업도 실행하지 않았다' in stream and 'No job has run' in stream
    assert started(log) == [] and not (workspace / 'work').exists() and not (workspace / 'dl2_db/db_dataset.grid.json').exists()


def test_one_remaining_fill_me_is_named_alone(etx_py, tmp_path, workspace):  # noqa: F811
    text = db_text(workspace, values={k: v for k, v in DB_VALUES.items() if k != 'TOPK'})
    proc = execute(etx_py, place(tmp_path, text, 'runner.py'), tmp_path)
    assert proc.returncode == 2 and 'ERROR edit_value_required: TOPK still ' in proc.err and 'CONFIG' not in proc.err.split('TOPK still')[0]
    assert started(read_log(workspace, DB_LOG)) == []


@pytest.mark.parametrize('which,label,remove', [
    ('db', 'LINEAGES', 'dl2_db/lineages.json'), ('db', 'DB_DATASET', 'dl2_db/db_dataset.json'), ('db', 'COHORT', 'dl2_db/cohort.json'),
    ('db', 'QUERY_DATASET', 'dl2_db/query_dataset.json'), ('eval', 'CANDIDATES[P2]', 'dl2_db/templates/P2/review_rois.json'),
    ('eval', 'CORRESPONDENCE', 'dl2_eval/correspondence.json'), ('eval', 'SEARCH_BATCH', 'work/db_out/search_batch.json'),
    ('eval', 'QUERY_EXPORT', 'dl2_db/templates/P0/review_rois.json')])
def test_a_missing_input_stops_before_any_job(etx_py, tmp_path, workspace, which, label, remove):  # noqa: F811
    (workspace / 'work/db_out').mkdir(parents=True)
    (workspace / 'work/db_out/search_batch.json').write_text('{}', encoding='utf-8')
    (workspace / remove).unlink()
    text = db_text(workspace) if which == 'db' else eval_text(workspace)
    proc = execute(etx_py, place(tmp_path, text, 'runner.py'), tmp_path)
    assert proc.returncode == 2 and 'Traceback' not in proc.err
    resolved = (workspace / remove).resolve()
    log = read_log(workspace, DB_LOG if which == 'db' else EVAL_LOG)
    for stream in (proc.err, log):
        assert f'ERROR input_missing: {label} -> {resolved} does not exist' in stream
        assert '이 입력 파일이 없다' in stream and '어떤 작업도 실행하지 않았다' in stream
    assert started(log) == []


def test_db_search_without_torch_stops_before_the_grid_is_made(etx_py, tmp_path, workspace):  # noqa: F811
    """adv:10: the torch check of the first torch job runs in _prepare, so no patches job and no grid manifest exist."""
    proc = execute(etx_py, place(tmp_path, db_text(workspace), 'runner.py'), tmp_path)
    assert proc.returncode == 2 and 'Traceback' not in proc.err
    log = read_log(workspace, DB_LOG)
    assert 'ERROR torch_required_for_make_split:' in log and 'ERROR torch_required_for_make_split:' in proc.err
    assert 'remove the make_split job from JOBS' in log and 'make_split 작업에는 torch 가 필요한데' in log
    for stream in (proc.err, log):   # T9 adv:1: before any job, "No job has run" replaces the per-job "earlier outputs are kept"
        assert 'No job has run' in stream and '어떤 작업도 실행하지 않았다' in stream
        assert 'Outputs of earlier jobs are kept' not in stream and '앞 작업의 산출물' not in stream
    assert started(log) == [] and not (workspace / 'work').exists() and not (workspace / 'dl2_db/db_dataset.grid.json').exists()


def test_product_count_mismatch_and_bad_names_stop_before_any_job(etx_py, tmp_path, workspace):  # noqa: F811
    text = edit(db_text(workspace), f'PRODUCTS = {PRODUCT_NAMES!r}', "PRODUCTS = ['P0', 'P1']")
    proc = execute(etx_py, place(tmp_path, text, 'runner.py'), tmp_path)
    assert proc.returncode == 2 and 'ERROR input_mismatch: PRODUCTS has 2 names but DB_DATASET' in proc.err and 'has 4 sources' in proc.err
    assert started(read_log(workspace, DB_LOG)) == []
    for bad in ("['P0', 'P0', 'P2', 'N']", "['P0', 'a/b', 'P2', 'N']"):
        text = edit(db_text(workspace), f'PRODUCTS = {PRODUCT_NAMES!r}', f'PRODUCTS = {bad}')
        proc = execute(etx_py, place(tmp_path, text, 'runner.py'), tmp_path)
        assert proc.returncode == 2 and 'ERROR edit_value_invalid: PRODUCTS must be' in proc.err, bad


def test_eval_without_ground_truth_runs_only_the_baseline_jobs(etx_py, tmp_path, workspace):  # noqa: F811
    """The EDIT block comment tells the user to write `JOBS = NCC_JOBS`; then no correspondence or search batch is needed."""
    text = eval_text(workspace, values={k: v for k, v in EVAL_VALUES.items() if k in ('WINDOW_UM', 'NCC_TOPK', 'MAX_MEMORY_MB')})
    text = edit(text, '\nJOBS = NCC_JOBS + EVAL_JOBS\n', '\nJOBS = NCC_JOBS\n')
    assert '`JOBS = NCC_JOBS`' in text
    (workspace / 'dl2_eval/correspondence.json').unlink()
    proc = execute(etx_py, place(tmp_path, text, 'runner.py'), tmp_path)
    assert proc.returncode == 0, proc.err
    log = read_log(workspace, EVAL_LOG)
    assert started(log) == ['baseline_ncc', 'baseline_ncc'] and 'without evaluation' in log
    assert not (workspace / 'work/eval_out/eval_baseline.json').exists()


# ---------------------------------------------------------------- T8: stricter pre-checks (no job runs)

class Raw:
    """A value written into the EDIT block as source text (nan, inf)."""

    def __init__(self, source, shown):
        self.source, self.shown = source, shown

    def __repr__(self):
        return self.source


def no_job_ran(workspace, which, proc, code, *fragments):
    assert proc.returncode == 2 and 'Traceback' not in proc.err
    log = read_log(workspace, DB_LOG if which == 'db' else EVAL_LOG)
    for stream in (proc.err, log):
        assert f'ERROR {code}: ' in stream
        for fragment in fragments:
            assert fragment in stream, fragment
    assert started(log) == [] and not (workspace / 'work').exists() and not (workspace / 'dl2_db/db_dataset.grid.json').exists()


NAN, INF = Raw('float("nan")', 'nan'), Raw('float("inf")', 'inf')


@pytest.mark.parametrize('which,name,value,shown', [
    ('db', 'TOPK', 0, '0'), ('db', 'TOPK', True, 'True'), ('db', 'TOPK', 2.5, '2.5'), ('db', 'TOPK', '4', "'4'"),
    ('db', 'PER_PRODUCT_TOPK', -1, '-1'), ('db', 'GRID_STEP_UM', 0, '0'), ('db', 'GRID_STEP_UM', NAN, 'nan'),
    ('db', 'GRID_STEP_UM', False, 'False'), ('db', 'DEVICE', 'gpu', "'gpu'"),
    ('eval', 'K', 0, '0'), ('eval', 'K', 1.5, '1.5'), ('eval', 'NCC_TOPK', -2, '-2'), ('eval', 'WINDOW_UM', -1, '-1'),
    ('eval', 'HIT_TOLERANCE_UM', 0, '0'), ('eval', 'MAX_MEMORY_MB', INF, 'inf'), ('eval', 'MAX_MEMORY_MB', NAN, 'nan')])
def test_a_wrongly_typed_edit_value_stops_before_any_job(etx_py, tmp_path, workspace, which, name, value, shown):  # noqa: F811
    base = DB_VALUES if which == 'db' else EVAL_VALUES
    values = dict(base, **{name: value})
    text = db_text(workspace, values=values) if which == 'db' else eval_text(workspace, values=values)
    proc = execute(etx_py, place(tmp_path, text, 'runner.py'), tmp_path)
    no_job_ran(workspace, which, proc, 'edit_value_invalid', f'{name}={shown} ', 'EDIT 블록', '어떤 작업도 실행하지 않았다', 'No job has run')


def test_an_invalid_device_names_the_way_to_pick_a_gpu(etx_py, tmp_path, workspace):  # noqa: F811
    """T9 adv:5."""
    proc = execute(etx_py, place(tmp_path, db_text(workspace, values=dict(DB_VALUES, DEVICE='gpu')), 'runner.py'), tmp_path)
    no_job_ran(workspace, 'db', proc, 'edit_value_invalid', "DEVICE='gpu' ", "DEVICE='cuda' and CUDA_VISIBLE_DEVICES", "DEVICE='cuda' 와 CUDA_VISIBLE_DEVICES 를 쓴다")


def test_per_product_topk_none_is_a_valid_value(etx_py, tmp_path, workspace):  # noqa: F811
    text = db_text(workspace, values=dict(DB_VALUES, PER_PRODUCT_TOPK=None))
    proc = execute(etx_py, place(tmp_path, text, 'runner.py'), tmp_path)
    assert 'edit_value_invalid' not in proc.err and 'ERROR torch_required_for_make_split' in proc.err   # it got past the value checks


@pytest.mark.parametrize('content,reason', [
    ('[]', 'must be a non-empty JSON array'), ('{"a": 1}', 'must be a non-empty JSON array'), ('["P0", "P0"]', 'entries must be distinct'),
    ('[1, "P1"]', 'every entry must be a non-empty string'), ('["P0", ""]', 'every entry must be a non-empty string'),
    ('["P0", "  "]', 'every entry must be a non-empty string'),
    ('not json', 'JSONDecodeError')])
def test_a_bad_cohort_file_stops_before_any_job(etx_py, tmp_path, workspace, content, reason):  # noqa: F811
    cohort = workspace / 'dl2_db/cohort.json'
    cohort.write_text(content, encoding='utf-8')
    proc = execute(etx_py, place(tmp_path, db_text(workspace), 'runner.py'), tmp_path)
    no_job_ran(workspace, 'db', proc, 'input_invalid', f'COHORT -> {cohort.resolve()}: ', reason, '어떤 작업도 실행하지 않았다')


def rewrite(path, change):
    document = json.loads(path.read_text(encoding='utf-8'))
    change(document)
    path.write_text(json.dumps(document), encoding='utf-8')


@pytest.mark.parametrize('label,file,change,reason', [
    ('QUERY_DATASET', 'query_dataset.json', lambda d: d.update(schema_version='psdl2_dataset_v2'), "schema_version is not 'psdl2_dataset_v1'"),
    ('QUERY_DATASET', 'query_dataset.json', lambda d: d['sources'][0].update(export_sha256='0' * 64), 'source 0: export_sha256 does not match'),
    ('QUERY_DATASET', 'query_dataset.json', lambda d: d['sources'][0].update(export='templates/none/review_rois.json'),
     'source 0: export file '),
    ('DB_DATASET', 'db_dataset.json', lambda d: d['sources'][1].update(export_sha256='f' * 64), 'source 1: export_sha256 does not match'),
    ('DB_DATASET', 'db_dataset.json', lambda d: d.update(sources=[]), 'sources must be a non-empty list'),
    ('DB_DATASET', 'db_dataset.json', lambda d: d.pop('schema_version'), "schema_version is not 'psdl2_dataset_v1'")])
def test_a_bad_dataset_manifest_stops_before_any_job(etx_py, tmp_path, workspace, label, file, change, reason):  # noqa: F811
    path = workspace / 'dl2_db' / file
    rewrite(path, change)
    proc = execute(etx_py, place(tmp_path, db_text(workspace), 'runner.py'), tmp_path)
    no_job_ran(workspace, 'db', proc, 'input_invalid', f'{label} -> {path.resolve()}: ', reason, '어떤 작업도 실행하지 않았다')


def test_a_db_dataset_that_already_has_patch_sets_stops_before_any_job(etx_py, tmp_path, workspace):  # noqa: F811
    rewrite(workspace / 'dl2_db/db_dataset.json', lambda d: d['sources'][2].update(patch_sets=['x/grid_P2.json']))
    proc = execute(etx_py, place(tmp_path, db_text(workspace), 'runner.py'), tmp_path)
    no_job_ran(workspace, 'db', proc, 'input_invalid', 'DB_DATASET source 2 already has patch_sets', 'patch_sets 가 없는 manifest')


def test_an_unwritable_grid_dataset_stops_before_any_job(torch_py, tmp_path, workspace):  # noqa: F811
    (workspace / 'dl2_db/db_dataset.grid.json').mkdir()   # a folder with the file name: the write fails even for root
    proc = execute(torch_py, place(tmp_path, db_text(workspace), 'runner.py'), tmp_path)
    grid = (workspace / 'dl2_db/db_dataset.grid.json').resolve()
    assert proc.returncode == 2 and 'Traceback' not in proc.err
    log = read_log(workspace, DB_LOG)
    for stream in (proc.err, log):
        assert f'ERROR output_unwritable: GRID_DATASET -> {grid} cannot be written' in stream
        assert 'Copy DB_DATASET and its exports to a folder you can write to' in stream and '쓸 수 있는 폴더로 복사' in stream
    assert started(log) == [] and not (workspace / 'work').exists()


def test_device_cuda_without_a_usable_cuda_stops_before_any_job(torch_py, tmp_path, workspace):  # noqa: F811
    proc = execute(torch_py, place(tmp_path, db_text(workspace, values=dict(DB_VALUES, DEVICE='cuda')), 'runner.py'), tmp_path)
    log = read_log(workspace, DB_LOG)
    if 'cuda available: False' not in log:   # a CUDA machine would train; this development host has none
        return
    no_job_ran(workspace, 'db', proc, 'cuda_unavailable', "DEVICE is 'cuda'", '"cpu" 로 바꾸거나', '어떤 작업도 실행하지 않았다')


def test_config_is_required_by_make_split_and_train_alone(etx_py, tmp_path, workspace):  # noqa: F811
    """adv:7: with only make_split and train left in JOBS, CONFIG is still checked (value, then file)."""
    def trimmed(values):
        text = db_text(workspace, values=values)
        marker = text.rindex('\n', 0, text.index('EDIT ABOVE')) + 1
        return text[:marker] + 'JOBS = [job for job in JOBS if job[0] in ("make_split", "train")]\n' + text[marker:]
    values = {k: v for k, v in DB_VALUES.items() if k in ('DEVICE',)}
    proc = execute(etx_py, place(tmp_path, trimmed(values), 'runner.py'), tmp_path)
    assert proc.returncode == 2 and 'ERROR edit_value_required: CONFIG still ' in proc.err
    assert 'GRID_STEP_UM' not in proc.err.split('CONFIG still')[0] and 'COHORT' not in proc.err.split('CONFIG still')[0]
    (workspace / 'dl2_db/config.json').unlink()
    proc = execute(etx_py, place(tmp_path, trimmed(dict(values, CONFIG='dl2_db/config.json')), 'runner.py'), tmp_path)
    assert proc.returncode == 2 and f'ERROR input_missing: CONFIG -> {(workspace / "dl2_db/config.json").resolve()} does not exist' in proc.err
    assert started(read_log(workspace, DB_LOG)) == []


def write_envelope(workspace, rankings):
    path = workspace / 'work/db_out/search_batch.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'schema_version': 'psdl2_search_batch_v1', 'rankings': rankings}), encoding='utf-8')
    return path


def test_an_envelope_of_another_query_export_stops_eval_before_any_job(etx_py, tmp_path, workspace):  # noqa: F811
    envelope = write_envelope(workspace, [{'query_roi_id': 'q', 'query_export_sha256': None},
                                          {'query_roi_id': 'r', 'query_export_sha256': 'ab' * 32}])
    query = workspace / 'dl2_db/templates/P0/review_rois.json'
    proc = execute(etx_py, place(tmp_path, eval_text(workspace), 'runner.py'), tmp_path)
    assert proc.returncode == 2 and 'Traceback' not in proc.err
    log = read_log(workspace, EVAL_LOG)
    for stream in (proc.err, log):
        assert 'ERROR input_mismatch: QUERY_EXPORT vs SEARCH_BATCH: ' in stream
        assert f'QUERY_EXPORT -> {query.resolve()} has sha256 {hashlib.sha256(query.read_bytes()).hexdigest()[:12]}' in stream
        assert f'SEARCH_BATCH -> {envelope.resolve()}' in stream and 'abababababab' in stream and '어떤 작업도 실행하지 않았다' in stream
    assert started(log) == [] and not (workspace / 'work/eval_out').exists()


def test_an_envelope_of_this_query_export_passes_and_one_without_hashes_is_not_checked(etx_py, tmp_path, workspace):  # noqa: F811
    query = workspace / 'dl2_db/templates/P0/review_rois.json'
    write_envelope(workspace, [{'query_roi_id': 'q', 'query_export_sha256': hashlib.sha256(query.read_bytes()).hexdigest()}])
    proc = execute(etx_py, place(tmp_path, eval_text(workspace), 'runner.py'), tmp_path)
    assert 'input_mismatch' not in proc.err and 'QUERY_EXPORT check' not in proc.err   # it reached the jobs (the toy envelope fails later)
    assert 'ranks' not in proc.err   # every query is from QUERY_EXPORT: no mixed-export line
    assert started(read_log(workspace, EVAL_LOG))[:2] == ['baseline_ncc', 'baseline_ncc']
    write_envelope(workspace, [{'query_roi_id': 'q', 'query_export_sha256': 'cd' * 32},   # a query dataset may span several exports
                               {'query_roi_id': 'r', 'query_export_sha256': hashlib.sha256(query.read_bytes()).hexdigest()}])
    proc = execute(etx_py, place(tmp_path, eval_text(workspace), 'runner.py'), tmp_path)
    assert 'input_mismatch' not in proc.err
    # T9 adv:2: a mixed envelope is reported, not refused
    assert 'prepare: SEARCH_BATCH ranks 1 of 2 queries from QUERY_EXPORT; 1 from other exports' in proc.err
    assert '질의 2 개 중 1 개가 QUERY_EXPORT 의 것이고 1 개는 다른 export 의 것이다' in proc.err
    write_envelope(workspace, [{'query_roi_id': 'q', 'query_export_sha256': None}])
    execute(etx_py, place(tmp_path, eval_text(workspace), 'runner.py'), tmp_path)
    assert 'records no query_export_sha256; skipping the QUERY_EXPORT check' in read_log(workspace, EVAL_LOG)


def test_an_unreadable_envelope_stops_eval_before_any_job(etx_py, tmp_path, workspace):  # noqa: F811
    (workspace / 'work/db_out').mkdir(parents=True)
    envelope = workspace / 'work/db_out/search_batch.json'
    envelope.write_text('{}', encoding='utf-8')
    proc = execute(etx_py, place(tmp_path, eval_text(workspace), 'runner.py'), tmp_path)
    assert proc.returncode == 2 and f'ERROR input_invalid: SEARCH_BATCH -> {envelope.resolve()}: ' in proc.err
    assert started(read_log(workspace, EVAL_LOG)) == []


@pytest.mark.parametrize('statuses,expected', [
    (['available', 'available', 'available'], 'search complete: 3 queries'),
    (['available', 'unavailable', 'unavailable'], 'search exit 3 (partial): 2 of 3 queries have no candidates')])
def test_the_search_summary_line_follows_the_envelope_status(workspace, monkeypatch, capsys, statuses, expected):
    text = DB.read_text(encoding='utf-8')
    namespace = {'__name__': 'psdl2_db_search_prefix'}
    exec(compile(text[:text.index('def run_jobs')], 'bundle_prefix', 'exec'), namespace)   # the helpers and the _check hook
    batch = {'rankings': [{'query_roi_id': f'q{i}', 'candidates': [{}] if s == 'available' else []} for i, s in enumerate(statuses)],
             'results': [{'status': s} for s in statuses]}
    (workspace / 'work/db_out').mkdir(parents=True)
    (workspace / 'work/db_out/search_batch.json').write_text(json.dumps(batch), encoding='utf-8')
    monkeypatch.chdir(workspace)
    monkeypatch.setitem(namespace, 'OUT_DIR', 'work/db_out')
    namespace['_check'](Path.cwd())
    lines = capsys.readouterr().err.strip().splitlines()
    assert lines[-1].endswith(expected) and 'search batch work/db_out/search_batch.json: 3 queries' in lines[-2]
