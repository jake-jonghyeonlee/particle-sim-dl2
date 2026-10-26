"""ETX runtime reproduction: Python 3.9.14, numpy 1.23.5, Pillow 10.1.0, no torch (scripts/setup_etx_venvs.sh).

The single-file runner is executed with `-I` by a separate interpreter, from another working directory, against a
BASE_DIR with a non-ASCII name that contains a space. Everything is synthetic and offline.

Interpreter: PSDL2_ETX_PY (default ~/.cache/psdl2-etx/venv39-notorch/bin/python). A missing interpreter is a FAILURE
(run `bash scripts/setup_etx_venvs.sh`); only PSDL2_ALLOW_MISSING_ETX_PY=1 turns it into a skip.
The py3.8 compile check uses PSDL2_PY38 (default: the uv cpython-3.8.20 interpreter) with the same rule.
"""
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zlib

import numpy as np
import pytest

import test_position_baseline as tp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
# D10: the committed ETX baseline_ncc bundle must carry the baseline_ncc import closure of this commit byte for byte,
# except the modules listed in D10_CHANGED_SINCE_BASELINE, which must equal the working tree instead.
# A Task that changes a closure module on purpose moves this constant to that Task's base commit and says why in the
# commit message. Tasks that touch other modules (search, index, train, ...) leave it alone.
# T3 (dl2-db/T3) moved it from efc6dac to 36ff7dc because it adds the multi-product mode to eval_position.py.
# T6 (dl2-db/T6) moved it to e0402d0 (its base commit) because it changes eval_position.py again (T3 review adv:2, adv:8).
D10_BASELINE_COMMIT = 'e0402d07f59bc1281a61579bee2e6153265fd803'
D10_CHANGED_SINCE_BASELINE = ('psdl2/eval_position.py',)
# The real D10 guarantee is separate: the baseline_ncc bundle of this commit writes the same output bytes as the bundle of
# this older commit for the existing single-search / single-baseline flow. It never moves.
D10_OUTPUT_COMMIT = 'efc6dac346c6e0fefb373cb850946ff377adb627'
DEFAULT_ETX_PY = Path('~/.cache/psdl2-etx/venv39-notorch/bin/python').expanduser()
DEFAULT_PY38 = Path('~/.local/share/uv/python/cpython-3.8.20-macos-aarch64-none/bin/python3').expanduser()
WORKSPACE = '작업 폴더 공백 ñ'
N = 60  # 30 um x 30 um at 0.5 um/px; the real JOBS use --window-um 20
MARKER = '.psdl2_done.json'


def _interpreter(env_name, default, setup_hint):
    path = Path(os.environ.get(env_name, str(default))).expanduser()
    if path.is_file():
        return path
    message = f'{env_name}: interpreter not found at {path}. {setup_hint}'
    if os.environ.get('PSDL2_ALLOW_MISSING_ETX_PY') == '1':
        pytest.skip(message)
    pytest.fail(message, pytrace=False)


@pytest.fixture(scope='module')
def etx_py():
    return _interpreter('PSDL2_ETX_PY', DEFAULT_ETX_PY, 'Create it with: bash scripts/setup_etx_venvs.sh '
                        '(set PSDL2_ALLOW_MISSING_ETX_PY=1 only to skip on purpose).')


@pytest.fixture(scope='module')
def py38():
    return _interpreter('PSDL2_PY38', DEFAULT_PY38, 'Install it with: uv python install 3.8 '
                        '(or set PSDL2_PY38; PSDL2_ALLOW_MISSING_ETX_PY=1 skips).')


@pytest.fixture(autouse=True)
def frame_size(monkeypatch):
    monkeypatch.setattr(tp, 'SHAPE', (N, N))


# ---------------------------------------------------------------- fixtures and helpers

def _masks():
    rng = np.random.default_rng(1)
    return [(rng.random((N, N)) < 0.35).astype(np.uint8) for _ in range(2)]


def make_workspace(base):
    """The four simcompare export folders the real JOBS read. B is A shifted by 6 rows and 4 columns (3 um in y, 2 um in x)."""
    e1, e2 = _masks()
    candidates = base / 'data/candidates_k3'
    query_rows = [('q1', 'A0', (15.0, 15.0), (e1, e2)), ('q2', 'A1', (18.0, 12.0), (e1, e2))]
    shifted_rows = [('cb', 'B1', (15.0, 15.0), (tp.shifted(e1, 6, 4), tp.shifted(e2, 6, 4)))]
    tp.write_export(candidates / 'reference_export', 'A', query_rows)
    tp.write_export(candidates / 'target_index_export', 'B', shifted_rows)
    tp.write_export(candidates / 'target_export', 'B', shifted_rows)
    tp.write_export(candidates / 'reference_index_export', 'A', query_rows)
    tp.correspondence(base / 'data/c.json', [('q1', (15.0, 15.0), (17.0, 18.0), 'ready')])
    return base


def ncc_job(name, query, index):
    return ('baseline_ncc', ['--all-ready', '--allow-unversioned-producer',
                             '--query-export', f'data/candidates_k3/{query}_export/review_rois.json',
                             '--index-export', f'data/candidates_k3/{index}_index_export/review_rois.json',
                             '--window-um', '20', '--topk', '4', '--max-memory-mb', '8000',
                             '--sheet-dir', f'data/dl2/{name}_sheet', '--out', f'data/dl2/{name}.json'], None)


R2T = ncc_job('r2t', 'reference', 'target')
T2R = ncc_job('t2r', 'target', 'reference')
EVAL = ('eval_position', ['--baseline', 'data/dl2/r2t.json', '--correspondence', 'data/c.json',
                          '--hit-tolerance-um', '0.5', '--k', '3', '--out', 'data/dl2/eval/position.json'], None)
FLOW = [R2T, T2R, EVAL]


def make_bundle(tmp_path, jobs, base_dir, *, log_file='runs/job.log', folder='upload dir'):
    sys.path.insert(0, str(ROOT / 'scripts'))
    try:
        from build_single_file import build
    finally:
        sys.path.pop(0)
    script = tmp_path / folder / 'job.py'
    build(script, jobs=jobs, log_file=log_file)
    text = script.read_text(encoding='utf-8').replace('BASE_DIR = None', f'BASE_DIR = {str(base_dir)!r}', 1)
    script.write_text(text, encoding='utf-8')
    return script


def execute(python, script, tmp_path, *, isolated=True, env=None):
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir(exist_ok=True)
    environment = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), **(env or {})}
    command = [str(python)] + (['-I'] if isolated else []) + [str(script)]
    proc = subprocess.run(command, capture_output=True, cwd=elsewhere, env=environment)
    proc.out = proc.stdout.decode('utf-8', 'replace')
    proc.err = proc.stderr.decode('utf-8', 'replace')
    return proc


def read_log(base, name='runs/job.log'):
    return (base / name).read_text(encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def workspace(tmp_path):
    return make_workspace(tmp_path / WORKSPACE)


# ---------------------------------------------------------------- full ETX flow

def test_full_flow_on_etx_python(etx_py, tmp_path, workspace):
    assert not (workspace / 'data/dl2').exists() and not (workspace / 'runs').exists()
    script = make_bundle(tmp_path, FLOW, workspace)
    proc = execute(etx_py, script, tmp_path)
    assert proc.returncode == 0, proc.err
    assert 'Traceback' not in proc.err
    # relative paths resolve from BASE_DIR, not from the launch folder or the runner folder; --out folders are created
    for name in ('r2t', 't2r'):
        assert (workspace / f'data/dl2/{name}.json').is_file()
        assert (workspace / f'data/dl2/{name}_sheet/sheet.html').is_file()
        assert list((workspace / f'data/dl2/{name}_sheet').glob('q*_query.png'))
    position = json.loads((workspace / 'data/dl2/eval/position.json').read_text(encoding='utf-8'))
    assert position['status'] == 'available' and position['summary']['top1_hit_rate'] == 1.0
    assert not (tmp_path / 'elsewhere/data').exists() and not (tmp_path / 'upload dir/data').exists()
    log = read_log(workspace)  # next to BASE_DIR, UTF-8, also holds what the console showed
    assert f'relative paths resolve from {workspace.resolve()} (BASE_DIR)' in log
    assert f'launched in {(tmp_path / "elsewhere").resolve()}' in log and str(script) in log
    assert 'python 3.9.' in log and 'numpy 1.23.5' in log and 'PIL 10.1.0' in log and 'torch unavailable' in log
    assert log.count(' start baseline_ncc ') == 2 and ' start eval_position ' in log
    assert 'baseline_ncc exit 0' in log and 'eval_position exit 0' in log


def test_done_marker_contents(etx_py, tmp_path, workspace):
    script = make_bundle(tmp_path, FLOW, workspace)
    assert execute(etx_py, script, tmp_path).returncode == 0
    marker = workspace / ('data/dl2/r2t.json' + MARKER)
    done = json.loads(marker.read_text(encoding='utf-8'))
    assert set(done) == {'module', 'runner_payload_sha256', 'argv', 'input_sha256s', 'outputs'}
    assert done['module'] == 'baseline_ncc' and done['argv'] == R2T[1]
    assert done['runner_payload_sha256'] == re.search(r'PAYLOAD_SHA256 = "([0-9a-f]{64})"', script.read_text(encoding='utf-8')).group(1)
    query, index = 'data/candidates_k3/reference_export/review_rois.json', 'data/candidates_k3/target_index_export/review_rois.json'
    assert done['input_sha256s'] == {query: sha(workspace / query), index: sha(workspace / index)}
    paths = [entry['path'] for entry in done['outputs']]
    assert paths == sorted(paths) and 'data/dl2/r2t.json' in paths and 'data/dl2/r2t_sheet/sheet.html' in paths
    assert any(p.startswith('data/dl2/r2t_sheet/q0001') and p.endswith('.png') for p in paths)
    assert all(entry['sha256'] == sha(workspace / entry['path']) for entry in done['outputs'])
    assert not list((workspace / 'data/dl2').glob('*.tmp*'))
    # eval_position writes a marker beside its --out as well, with its own input (the previous output) recorded
    evaluated = json.loads((workspace / ('data/dl2/eval/position.json' + MARKER)).read_text(encoding='utf-8'))
    assert evaluated['module'] == 'eval_position' and 'data/dl2/r2t.json' in evaluated['input_sha256s']
    assert [e['path'] for e in evaluated['outputs']] == ['data/dl2/eval/position.json']


# ---------------------------------------------------------------- Python / numpy differences against the dev interpreter

FLOAT_ABS_TOL = 1e-9


def assert_json_close(left, right, where='$'):
    """Everything except floats must be identical. Floats may differ by FLOAT_ABS_TOL: numpy 1.23.5 and 2.x use different FFT
    back ends, so ZNCC scores differ at the 1e-16 level (observed: 0.9999999999999996 vs 1.0). Ranks, coordinates and counts do not move."""
    if isinstance(left, float) and isinstance(right, float):
        assert abs(left - right) <= FLOAT_ABS_TOL, f'{where}: {left!r} != {right!r}'
    elif isinstance(left, dict) and isinstance(right, dict):
        assert list(left) == list(right), f'{where}: keys differ'
        for key in left:
            assert_json_close(left[key], right[key], f'{where}.{key}')
    elif isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right), f'{where}: length differs'
        for index, (a, b) in enumerate(zip(left, right)):
            assert_json_close(a, b, f'{where}[{index}]')
    else:
        assert type(left) is type(right) and left == right, f'{where}: {left!r} != {right!r}'


def test_etx_python_output_matches_dev_python(etx_py, tmp_path):
    outputs = {}
    for label, python in (('etx', etx_py), ('dev', Path(sys.executable))):
        sub = tmp_path / label
        sub.mkdir()
        base = make_workspace(sub / WORKSPACE)
        script = make_bundle(sub, FLOW, base)
        proc = execute(python, script, sub)
        assert proc.returncode == 0, proc.err
        outputs[label] = base / 'data/dl2'
    etx, dev = outputs['etx'], outputs['dev']
    names = sorted(p.relative_to(etx).as_posix() for p in etx.rglob('*') if p.is_file() and not p.name.endswith(MARKER))
    assert names == sorted(p.relative_to(dev).as_posix() for p in dev.rglob('*') if p.is_file() and not p.name.endswith(MARKER))
    assert names
    from PIL import Image
    for name in names:
        if name.endswith('.json'):
            assert_json_close(json.loads((etx / name).read_text(encoding='utf-8')), json.loads((dev / name).read_text(encoding='utf-8')), name)
        elif name.endswith('.png'):  # PNG bytes depend on the Pillow/zlib build; the pixels must not
            with Image.open(etx / name) as a, Image.open(dev / name) as b:
                assert a.size == b.size and np.array_equal(np.asarray(a), np.asarray(b)), name
        else:
            assert (etx / name).read_bytes() == (dev / name).read_bytes(), name


# ---------------------------------------------------------------- console encoding

def test_ascii_console_with_non_ascii_base_dir_keeps_a_utf8_log(etx_py, tmp_path, workspace):
    script = make_bundle(tmp_path, [R2T], workspace)
    # (a) isolated mode with the C locale, (b) a strict ASCII console (PYTHONIOENCODING is ignored by -I, so no -I here)
    for isolated, env in ((True, {'LC_ALL': 'C'}), (False, {'LC_ALL': 'C', 'PYTHONIOENCODING': 'ascii:strict'})):
        shutil.rmtree(workspace / 'data/dl2', ignore_errors=True)
        if (workspace / 'runs/job.log').exists():
            (workspace / 'runs/job.log').unlink()
        proc = execute(etx_py, script, tmp_path, isolated=isolated, env=env)
        assert proc.returncode == 0, proc.err
        if not isolated:
            proc.stderr.decode('ascii')  # raises if a non-ASCII byte reached the console
            proc.stdout.decode('ascii')
        assert WORKSPACE in read_log(workspace)
        assert (workspace / 'data/dl2/r2t.json').is_file()
        if not isolated:
            assert 'relative paths resolve from' in proc.err and WORKSPACE not in proc.err


def test_ascii_console_without_a_log_file(etx_py, tmp_path, workspace):
    script = make_bundle(tmp_path, [R2T], workspace)
    text = script.read_text(encoding='utf-8')
    script.write_text(re.sub(r'(?m)^LOG_FILE = .*$', 'LOG_FILE = None', text, count=1), encoding='utf-8')
    proc = execute(etx_py, script, tmp_path, isolated=False, env={'LC_ALL': 'C', 'PYTHONIOENCODING': 'ascii:strict'})
    assert proc.returncode == 0, proc.err
    assert 'Traceback' not in proc.err
    proc.stderr.decode('ascii')
    proc.stdout.decode('ascii')
    assert 'relative paths resolve from' in proc.err and WORKSPACE not in proc.err
    assert (workspace / 'data/dl2/r2t.json').is_file() and not (workspace / 'runs').exists()


def test_tee_writes_the_log_first_and_survives_a_strict_ascii_console(tmp_path, workspace):
    import io
    import runpy
    script = make_bundle(tmp_path, [R2T], workspace)
    namespace = runpy.run_path(str(script), run_name='psdl2_bundle_under_test')  # defines everything, runs nothing
    events = []

    class Log(io.StringIO):
        def write(self, text):
            events.append('log')
            return super().write(text)

    log, raw = Log(), io.BytesIO()
    console = io.TextIOWrapper(raw, encoding='ascii', errors='strict')
    tee = namespace['_Tee'](log, console)
    original_write = console.write
    console.write = lambda text: (events.append('console'), original_write(text))[1]
    assert tee.write(f'경로 {WORKSPACE}\n') > 0
    tee.flush()
    assert events[:2] == ['log', 'console']
    assert log.getvalue() == f'경로 {WORKSPACE}\n'
    assert raw.getvalue().decode('ascii').startswith('?? ')


# ---------------------------------------------------------------- error codes

def test_base_dir_missing(etx_py, tmp_path):
    script = make_bundle(tmp_path, [R2T], tmp_path / '없는 폴더')
    for isolated, env in ((True, None), (False, {'PYTHONIOENCODING': 'ascii:strict'})):
        proc = execute(etx_py, script, tmp_path, isolated=isolated, env=env)
        assert proc.returncode == 2
        assert 'ERROR base_dir_missing:' in proc.err and 'BASE_DIR' in proc.err and 'absolute path of an existing folder' in proc.err
        assert isolated is False or '절대 경로' in proc.err  # an ASCII console keeps the English half, a UTF-8 one shows both
        assert 'Traceback' not in proc.err
    assert not (tmp_path / '없는 폴더').exists() and not (tmp_path / 'elsewhere/runs').exists()


def test_log_file_unwritable(etx_py, tmp_path, workspace):
    (workspace / 'blocker').write_text('a file where the log folder should be', encoding='utf-8')
    script = make_bundle(tmp_path, [R2T], workspace, log_file='blocker/job.log')
    proc = execute(etx_py, script, tmp_path)
    assert proc.returncode == 2
    assert 'ERROR log_file_unwritable:' in proc.err and 'LOG_FILE' in proc.err and 'None' in proc.err and '쓸 수 있는 경로' in proc.err
    assert 'Traceback' not in proc.err and not (workspace / 'data/dl2').exists()


@pytest.mark.parametrize('module', ['train', 'index', 'search', 'evaluate', 'make_split'])
def test_torch_required_stops_with_a_code_and_keeps_earlier_outputs(etx_py, tmp_path, workspace, module):
    script = make_bundle(tmp_path, [R2T, (module, ['--help'], None)], workspace)
    proc = execute(etx_py, script, tmp_path)
    assert proc.returncode == 2
    assert f'ERROR torch_required_for_{module}:' in proc.err and 'torch' in proc.err and 'JOBS' in proc.err
    assert 'Traceback' not in proc.err and f'start {module}' not in proc.err
    assert (workspace / 'data/dl2/r2t.json').is_file() and (workspace / ('data/dl2/r2t.json' + MARKER)).is_file()
    assert 'ERROR torch_required_for_' in read_log(workspace)


def test_torch_import_failed_is_distinguished_and_imports_torch_once(etx_py, tmp_path, workspace):
    script = make_bundle(tmp_path, [('search', ['--help'], None)], workspace)
    digest = re.search(r'PAYLOAD_SHA256 = "([0-9a-f]{64})"', script.read_text(encoding='utf-8')).group(1)
    package_root = script.parent / f'.psdl2_pkg_{digest[:12]}'  # the runner puts this folder first on sys.path
    package_root.mkdir()
    counter = tmp_path / 'torch_imports.txt'
    (package_root / 'torch.py').write_text(
        'import os\nopen(os.environ["TORCH_COUNT_FILE"], "a").write("x")\nraise ImportError("broken torch for the test")\n', encoding='utf-8')
    proc = execute(etx_py, script, tmp_path, env={'TORCH_COUNT_FILE': str(counter)})
    assert proc.returncode == 2
    # the line carries the exception type and message of the first import, and the check does not import again
    assert 'ERROR torch_import_failed: ImportError: broken torch for the test.' in proc.err and 'torch_required_for' not in proc.err
    assert 'Traceback' not in proc.err
    assert counter.read_text() == 'x'


@pytest.fixture
def completed(etx_py, tmp_path, workspace):
    script = make_bundle(tmp_path, FLOW, workspace)
    assert execute(etx_py, script, tmp_path).returncode == 0
    return script, workspace


def test_restart_skips_completed_jobs_and_resumes_at_the_failed_one(etx_py, tmp_path, workspace):
    (workspace / 'data/c.json').rename(workspace / 'data/c.hidden')  # job 3 fails: correspondence file is missing
    script = make_bundle(tmp_path, FLOW[:1] + FLOW[2:], workspace)
    first = execute(etx_py, script, tmp_path)
    assert first.returncode == 2
    assert (workspace / ('data/dl2/r2t.json' + MARKER)).is_file()
    assert not (workspace / ('data/dl2/eval/position.json' + MARKER)).exists()
    before = (workspace / 'data/dl2/r2t.json').read_bytes()
    (workspace / 'data/c.hidden').rename(workspace / 'data/c.json')
    second = execute(etx_py, script, tmp_path)
    assert second.returncode == 0, second.err
    log = read_log(workspace)
    assert log.count(' start baseline_ncc ') == 1 and 'skip completed baseline_ncc' in log
    assert log.count(' start eval_position ') == 2
    assert (workspace / 'data/dl2/r2t.json').read_bytes() == before
    third = execute(etx_py, script, tmp_path)  # everything is complete now
    assert third.returncode == 0 and 'skip completed eval_position' in read_log(workspace)
    assert read_log(workspace).count(' start eval_position ') == 2


def _damage_output(base, marker):
    with open(base / 'data/dl2/r2t.json', 'ab') as handle:
        handle.write(b' ')


def _remove_output(base, marker):
    sorted((base / 'data/dl2/r2t_sheet').glob('*.png'))[0].unlink()


def _change_input(base, marker):
    with open(base / 'data/candidates_k3/reference_export/review_rois.json', 'ab') as handle:
        handle.write(b'\n')


def _other_runner(base, marker):
    done = json.loads(marker.read_text(encoding='utf-8'))
    done['runner_payload_sha256'] = '0' * 64
    marker.write_text(json.dumps(done), encoding='utf-8')


@pytest.mark.parametrize('reason,damage', [('output_modified', _damage_output), ('output_missing', _remove_output),
                                           ('inputs_changed', _change_input), ('runner_changed', _other_runner)])
def test_stale_done_marker(etx_py, tmp_path, completed, reason, damage):
    script, base = completed
    damage(base, base / ('data/dl2/r2t.json' + MARKER))
    starts = read_log(base).count(' start ')
    proc = execute(etx_py, script, tmp_path)
    assert proc.returncode == 2
    assert f'ERROR stale_done_marker:{reason}:' in proc.err
    assert '표지' in proc.err and '지우고 다시 실행' in proc.err and 'r2t.json' + MARKER in proc.err
    assert 'Traceback' not in proc.err
    log = read_log(base)
    assert log.count(' start ') == starts and 'skip completed' not in log  # nothing was re-run or skipped silently


def _truncate(marker):
    marker.write_text('{"module": "baseline_ncc", "argv": [', encoding='utf-8')


def _drop_key(marker):
    done = json.loads(marker.read_text(encoding='utf-8'))
    del done['input_sha256s']
    marker.write_text(json.dumps(done), encoding='utf-8')


def _wrong_type(marker):
    done = json.loads(marker.read_text(encoding='utf-8'))
    done['outputs'] = 'not a list'
    marker.write_text(json.dumps(done), encoding='utf-8')


def _wrong_entry(marker):
    done = json.loads(marker.read_text(encoding='utf-8'))
    done['outputs'] = [{'path': 5}]
    marker.write_text(json.dumps(done), encoding='utf-8')


@pytest.mark.parametrize('damage', [_truncate, _drop_key, _wrong_type, _wrong_entry])
def test_unreadable_or_misshapen_marker_stops_with_a_code_not_a_traceback(etx_py, tmp_path, completed, damage):
    script, base = completed
    damage(base / ('data/dl2/r2t.json' + MARKER))
    proc = execute(etx_py, script, tmp_path)
    assert proc.returncode == 2
    assert 'ERROR stale_done_marker:marker_unreadable:' in proc.err and 'Traceback' not in proc.err


def test_zero_exit_without_the_out_file_writes_no_marker(etx_py, tmp_path, workspace):
    job = ('eval_position', ['--help', '--out', 'data/dl2/never_written.json'], None)  # --help exits 0 before any output
    script = make_bundle(tmp_path, [job], workspace)
    proc = execute(etx_py, script, tmp_path)
    assert proc.returncode == 2
    assert 'ERROR job_output_missing:' in proc.err and 'Traceback' not in proc.err
    assert not list((workspace / 'data').rglob('never_written*'))


def test_out_ending_with_a_separator_is_rejected(etx_py, tmp_path, workspace):
    job = ('baseline_ncc', [*R2T[1][:-1], 'data/dl2/folder/'], None)
    script = make_bundle(tmp_path, [job], workspace)
    proc = execute(etx_py, script, tmp_path)
    assert proc.returncode == 2
    assert 'ERROR out_must_be_file_path:' in proc.err and 'Traceback' not in proc.err and 'start baseline_ncc' not in proc.err
    assert not (workspace / 'data/dl2').exists()


def test_modules_outside_the_whitelist_write_and_read_no_marker_and_say_so(etx_py, tmp_path, workspace):
    out = workspace / 'data/dl2/patches_help.json'
    out.parent.mkdir(parents=True)
    stale = Path(str(out) + MARKER)
    stale.write_text('this is not JSON and must never be read', encoding='utf-8')
    script = make_bundle(tmp_path, [('patches', ['--help', '--out', 'data/dl2/patches_help.json'], None)], workspace)
    for _ in range(2):
        proc = execute(etx_py, script, tmp_path)
        assert proc.returncode == 0, proc.err
    log = read_log(workspace)
    assert log.count('no done marker for patches (indirect inputs are not hashed)') == 2
    assert log.count(' start patches ') == 2 and 'skip completed' not in log
    assert stale.read_text(encoding='utf-8') == 'this is not JSON and must never be read'
    # the two whitelisted modules still get one
    assert 'no done marker for baseline_ncc' not in log


# ---------------------------------------------------------------- torch smoke bundle (X9)

DEFAULT_TORCH_PY = Path('~/.cache/psdl2-etx/venv39-torch/bin/python').expanduser()
SMOKE_LOG = 'psdl2_smoke/smoke.log'


@pytest.fixture(scope='module')
def torch_py():
    return _interpreter('PSDL2_ETX_TORCH_PY', DEFAULT_TORCH_PY, 'Create it with: bash scripts/setup_etx_venvs.sh '
                        '(set PSDL2_ALLOW_MISSING_ETX_PY=1 only to skip on purpose).')


def build_smoke_bundle(tmp_path, base_dir, *, folder='upload dir'):
    sys.path.insert(0, str(ROOT / 'scripts'))
    try:
        from build_etx_smoke import build_smoke
    finally:
        sys.path.pop(0)
    script = tmp_path / folder / 'smoke.py'
    build_smoke(script)
    script.write_text(script.read_text(encoding='utf-8').replace('BASE_DIR = None', f'BASE_DIR = {str(base_dir)!r}', 1), encoding='utf-8')
    return script


def test_smoke_bundle_passes_on_the_etx_torch_python(torch_py, tmp_path):
    base = tmp_path / WORKSPACE
    base.mkdir()
    script = build_smoke_bundle(tmp_path, base)
    proc = execute(torch_py, script, tmp_path)  # `-I`, another cwd, a BASE_DIR with a space and non-ASCII characters
    assert proc.returncode == 0, proc.err
    assert 'Traceback' not in proc.err
    log = read_log(base, SMOKE_LOG)
    assert 'python 3.9.' in log and 'numpy 1.23.5' in log and 'PIL 10.1.0' in log and 'torch 2.1.1' in log and 'cuda available:' in log
    assert re.search(r'setuptools (\d|unavailable)', log) and 'cuda check failed' not in log
    for step in ('patches', 'make_split', 'train', 'index', 'search', 'eval_position'):
        assert f' start {step} ' in log and f' {step} exit 0 in ' in log
    assert log.count(' start patches ') == 8 and 'smoke check OK' in log and 'eval check OK' in log
    evaluation = json.loads((base / 'psdl2_smoke/out/eval_position.json').read_text(encoding='utf-8'))
    assert evaluation['schema_version'] == 'psdl2_eval_position_multi_v1' and evaluation['summary']['n_queries_with_truth'] == 3
    assert evaluation['arguments']['mode'] == 'search_batch' and evaluation['status'] == 'available'
    batch = json.loads((base / 'psdl2_smoke/out/search_batch.json').read_text(encoding='utf-8'))
    assert batch['schema_version'] == 'psdl2_search_batch_v1' and len(batch['results']) == 3
    for ranking in batch['rankings']:
        products = {c['product_id'] for c in ranking['candidates']}
        assert ranking['candidates'] and ranking['query_product_id'] not in products and 'N' not in products
    assert not (tmp_path / 'elsewhere/psdl2_smoke').exists()
    # a second run reuses the same folder and ends the same way
    again = execute(torch_py, script, tmp_path)
    assert again.returncode == 0, again.err
    assert json.loads((base / 'psdl2_smoke/out/search_batch.json').read_text(encoding='utf-8'))['rankings'] == batch['rankings']


def test_version_line_has_setuptools_and_a_failed_cuda_check_keeps_torch_usable(etx_py, tmp_path, workspace):
    script = make_bundle(tmp_path, [('search', ['--help'], None)], workspace)
    assert 'for optional in ("PIL", "setuptools", "torch"):' in script.read_text(encoding='utf-8')
    digest = re.search(r'PAYLOAD_SHA256 = "([0-9a-f]{64})"', script.read_text(encoding='utf-8')).group(1)
    package_root = script.parent / f'.psdl2_pkg_{digest[:12]}'  # the runner puts this folder first on sys.path
    package_root.mkdir()
    (package_root / 'torch.py').write_text(
        '__version__ = "9.9"\n\n\nclass cuda:\n    @staticmethod\n    def is_available():\n        raise RuntimeError("no driver")\n',
        encoding='utf-8')
    proc = execute(etx_py, script, tmp_path)
    assert proc.returncode == 0, proc.err
    log = read_log(workspace)
    assert re.search(r'setuptools (\d|unavailable)', log) and 'torch 9.9' in log
    assert 'cuda check failed: RuntimeError' in log and 'cuda available' not in log
    assert 'ERROR' not in log and ' search exit 0 ' in log  # the job still ran: torch counts as usable


def test_smoke_bundle_stops_at_the_first_torch_job_without_torch(etx_py, tmp_path):
    base = tmp_path / WORKSPACE
    base.mkdir()
    script = build_smoke_bundle(tmp_path, base)
    proc = execute(etx_py, script, tmp_path)
    assert proc.returncode == 2
    assert 'ERROR torch_required_for_make_split:' in proc.err and 'Traceback' not in proc.err
    log = read_log(base, SMOKE_LOG)
    assert log.count(' patches exit 0 ') == 8 and ' start make_split ' not in log  # numpy-only jobs ran, the first torch job did not


def test_smoke_bundle_build_is_deterministic_and_its_views_match_psdl2(tmp_path):
    from dataclasses import asdict
    sys.path.insert(0, str(ROOT / 'scripts'))
    try:
        import build_etx_smoke
    finally:
        sys.path.pop(0)
    out = tmp_path / 'again.py'
    build_etx_smoke.main(['--out', str(out)])
    assert out.read_bytes() == SMOKE.read_bytes()
    from test_views import view_config
    assert json.loads(json.dumps(asdict(view_config(jitter=(.1, .1))))) == json.loads(json.dumps(build_etx_smoke.VIEWS))


# ---------------------------------------------------------------- bundle equals the efc6dac bundle (D10)

def git_show(path, commit):
    proc = subprocess.run(['git', 'show', f'{commit}:{path}'], capture_output=True, cwd=ROOT)
    if proc.returncode != 0:
        message = f'commit {commit[:7]} is not reachable from this checkout (git show failed)'
        if os.environ.get('PSDL2_ALLOW_MISSING_ETX_PY') == '1':
            pytest.skip(message)
        pytest.fail(message + '. Fetch the history, or set PSDL2_ALLOW_MISSING_ETX_PY=1 to skip on purpose.', pytrace=False)
    return proc.stdout


def payload_files(bundle_text):
    tree = ast.parse(bundle_text)
    literal = next(node.value for node in tree.body
                   if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'PAYLOAD' for t in node.targets))
    packed = ast.literal_eval(literal)
    packed = packed if isinstance(packed, str) else ''.join(packed)
    return {name: base64.b64decode(entry['data']) for name, entry in json.loads(zlib.decompress(base64.b64decode(packed))).items()}


def import_closure(files, start='psdl2/baseline_ncc.py'):
    seen, todo = set(), [start]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(ast.parse(files[name])):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                targets = [node.module] if node.module else [alias.name for alias in node.names]
                todo += [f'psdl2/{t}.py' for t in targets if f'psdl2/{t}.py' in files]
    return seen


def edit_section(text):
    start = text.index('"""\n', text.index('"""') + 3) + 4  # after the docstring: header comments, EDIT block with JOBS
    return text[start:text.index('\n', text.index('EDIT ABOVE'))]


COMMITTED = ROOT / 'bundles/etx/psdl2_job_baseline_ncc.py'
SMOKE = ROOT / 'bundles/etx/psdl2_job_smoke_torch.py'


def test_regenerated_bundle_payload_is_the_baseline_commit_source():
    files = payload_files(COMMITTED.read_text(encoding='utf-8'))
    closure = import_closure(files)
    assert {'psdl2/baseline_ncc.py', 'psdl2/data.py', 'psdl2/eval_position.py', 'psdl2/schema.py'} <= closure
    # search, index and the torch modules are outside the closure, so a Task that only changes them keeps the constant
    assert not closure & {'psdl2/search.py', 'psdl2/index.py', 'psdl2/train.py', 'psdl2/model.py'}
    assert set(D10_CHANGED_SINCE_BASELINE) <= closure
    for name in sorted(closure):
        if name in D10_CHANGED_SINCE_BASELINE:
            assert files[name] == (SRC / name).read_bytes(), f'{name} differs from the working tree'
            assert files[name] != git_show(f'src/{name}', D10_BASELINE_COMMIT), f'{name} is unchanged: drop it from the list'
        else:
            assert files[name] == git_show(f'src/{name}', D10_BASELINE_COMMIT), f'{name} differs from {D10_BASELINE_COMMIT[:7]}'


def test_regenerated_bundle_keeps_the_edit_block_and_jobs():
    old = git_show('bundles/etx/psdl2_job_baseline_ncc.py', D10_OUTPUT_COMMIT).decode('utf-8')
    new = COMMITTED.read_text(encoding='utf-8')
    assert 'JOBS = [' in edit_section(new) and edit_section(new) == edit_section(old)


def test_regenerated_bundle_output_is_byte_identical_to_efc6dac_bundle(etx_py, tmp_path):
    assert D10_OUTPUT_COMMIT.startswith('efc6dac')
    """No run-specific field (time, path) appears in the JSON or sheet, so nothing is excluded; the new done markers are
    the only extra files and are not part of the comparison."""
    trees = {}
    for label in ('old', 'new'):
        sub = tmp_path / label
        sub.mkdir()
        base = make_workspace(sub / WORKSPACE)
        script = sub / 'upload dir/job.py'
        script.parent.mkdir()
        text = (git_show('bundles/etx/psdl2_job_baseline_ncc.py', D10_OUTPUT_COMMIT).decode('utf-8') if label == 'old'
                else COMMITTED.read_text(encoding='utf-8'))
        script.write_text(text.replace('BASE_DIR = None', f'BASE_DIR = {str(base)!r}', 1), encoding='utf-8')
        proc = execute(etx_py, script, sub)
        assert proc.returncode == 0, proc.err
        trees[label] = {p.relative_to(base / 'data/dl2').as_posix(): p.read_bytes() for p in (base / 'data/dl2').rglob('*')
                        if p.is_file() and not p.name.endswith(MARKER)}
    assert trees['old'] and trees['old'] == trees['new']


def test_etx_bundle_build_is_deterministic(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    try:
        import build_etx_runner
    finally:
        sys.path.pop(0)
    out = tmp_path / 'again.py'
    build_etx_runner.main(['--out', str(out)])
    assert out.read_bytes() == COMMITTED.read_bytes()


# ---------------------------------------------------------------- static checks for Python 3.9 / 3.8

def _sources():
    return sorted((SRC / 'psdl2').glob('*.py'))


COMPILE_ALL = 'import sys\nfor p in sys.argv[1:]:\n    compile(open(p, encoding="utf-8").read(), p, "exec")\nprint("compiled", len(sys.argv) - 1)\n'


def _compile_with(python, files):
    proc = subprocess.run([str(python), '-I', '-c', COMPILE_ALL, *map(str, files)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert f'compiled {len(files)}' in proc.stdout


def test_sources_and_bundle_compile_on_python39(etx_py, tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    try:
        from build_single_file import build
    finally:
        sys.path.pop(0)
    generated = tmp_path / 'generated.py'
    build(generated)
    _compile_with(etx_py, [*_sources(), generated, COMMITTED, SMOKE, ROOT / 'scripts/build_single_file.py', ROOT / 'scripts/build_etx_runner.py',
                           ROOT / 'scripts/build_etx_smoke.py'])


def test_sources_and_bundle_still_compile_on_python38(py38):
    _compile_with(py38, [*_sources(), COMMITTED, SMOKE])


def test_modules_import_on_python39_without_torch(etx_py):
    needs_torch = {'checkpoint', 'evaluate', 'index', 'losses', 'make_split', 'model', 'train'}
    code = ('import sys, importlib\nsys.path.insert(0, sys.argv[1])\n'
            'try:\n    importlib.import_module("psdl2." + sys.argv[2])\nexcept ModuleNotFoundError as exc:\n'
            '    print("MNF", exc.name)\nelse:\n    print("OK")\n')
    for path in _sources():
        if path.stem == '__init__':
            continue
        proc = subprocess.run([str(etx_py), '-I', '-c', code, str(SRC), path.stem], capture_output=True, text=True)
        assert proc.returncode == 0, f'{path.stem}: {proc.stderr}'
        assert proc.stdout.strip() == ('MNF torch' if path.stem in needs_torch else 'OK'), path.stem


BUILTIN_TYPES = {'int', 'str', 'float', 'bool', 'bytes', 'list', 'dict', 'tuple', 'set', 'frozenset', 'object'}
BANNED_CALLS = {'aiter', 'anext'}          # Python 3.10+
BANNED_ATTRIBUTES = {'bit_count', 'pairwise'}  # int.bit_count, itertools.pairwise: 3.10+
BANNED_KEYWORDS = {('zip', 'strict'), ('dataclass', 'slots'), ('dataclass', 'kw_only'), ('dataclass', 'match_args')}
# str.removeprefix/removesuffix, math.ulp, Path.is_relative_to and dict | dict are 3.9 features and are allowed.


def python310_findings(path, source):
    tree = ast.parse(source)
    skip = set()
    if any(isinstance(n, ast.ImportFrom) and n.module == '__future__' and any(a.name == 'annotations' for a in n.names)
           for n in tree.body):  # annotations are never evaluated, so `int | None` there is fine
        for node in ast.walk(tree):
            annotations = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                annotations = [node.returns] + [a.annotation for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs]
                annotations += [node.args.vararg and node.args.vararg.annotation, node.args.kwarg and node.args.kwarg.annotation]
            elif isinstance(node, ast.AnnAssign):
                annotations = [node.annotation]
            for annotation in filter(None, annotations):
                skip.update(id(n) for n in ast.walk(annotation))
    findings = []
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, (ast.Match, getattr(ast, 'TryStar', ast.Match))):
            findings.append((node.lineno, 'match/except* statement'))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            for side in (node.left, node.right):
                if (isinstance(side, ast.Name) and side.id in BUILTIN_TYPES) or (isinstance(side, ast.Constant) and side.value is None):
                    findings.append((node.lineno, 'X | Y type union at runtime'))
                    break
        elif isinstance(node, ast.ImportFrom) and any(a.name in BANNED_ATTRIBUTES for a in node.names):
            findings.append((node.lineno, '.pairwise'))
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRIBUTES:
            findings.append((node.lineno, f'.{node.attr}'))
        elif isinstance(node, ast.Call):
            func = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', None)
            if func in BANNED_CALLS:
                findings.append((node.lineno, f'{func}()'))
            findings += [(node.lineno, f'{func}({k.arg}=)') for k in node.keywords if (func, k.arg) in BANNED_KEYWORDS]
    findings += [(number, 'parenthesized with-statement') for number, line in enumerate(source.splitlines(), 1)
                 if re.match(r'\s*(async\s+)?with\s*\((?:[^)]*|.*\bas\b.*\)\s*:)$', line)]
    return findings


def test_no_python310_syntax_or_api_in_sources_and_bundle():
    paths = [*_sources(), COMMITTED, SMOKE, ROOT / 'bundles/etx/psdl2_job_db_search.py', ROOT / 'bundles/etx/psdl2_job_eval.py',
             ROOT / 'scripts/build_single_file.py', ROOT / 'scripts/build_etx_runner.py', ROOT / 'scripts/build_etx_smoke.py',
             ROOT / 'scripts/build_etx_jobs.py']
    found = {str(p.relative_to(ROOT)): python310_findings(p, p.read_text(encoding='utf-8')) for p in paths}
    assert {k: v for k, v in found.items() if v} == {}
    # the checker must catch what it claims to
    sample = ('from itertools import pairwise\nx = int | None\nwith (open("a") as f,\n      open("b") as g):\n    pass\n'
              'zip([], [], strict=True)\nn.bit_count()\n')
    assert {kind for _, kind in python310_findings('sample', sample)} >= {
        'X | Y type union at runtime', 'zip(strict=)', '.bit_count', '.pairwise'}
    assert python310_findings('ok', 'from __future__ import annotations\ndef f(a: int | None) -> str | None:\n    return a\n') == []
