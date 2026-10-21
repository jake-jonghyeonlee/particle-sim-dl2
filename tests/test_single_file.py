import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _build(tmp_path, jobs, base_dir=None):
    sys.path.insert(0, str(ROOT / 'scripts'))
    try:
        from build_single_file import build
    finally:
        sys.path.pop(0)
    out = tmp_path / 'job.py'
    build(out, jobs=[(name, args, None) for name, args in jobs], log_file='runs/job.log')
    if base_dir is not None:
        text = out.read_text(encoding='utf-8').replace('BASE_DIR = None', f'BASE_DIR = {str(base_dir)!r}', 1)
        out.write_text(text, encoding='utf-8')
    return out


def test_runner_extracts_bundle_logs_versions_and_stops_on_error(tmp_path):
    # query export가 없으므로 baseline_ncc는 종료 코드 2이고, 다음 작업은 실행되지 않는다.
    jobs = [('baseline_ncc', ['--all-ready', '--query-export', 'missing.json', '--index-export', 'missing.json',
                              '--window-um', '20', '--topk', '4', '--max-memory-mb', '100', '--out', 'out.json']),
            ('eval_position', ['--help'])]
    script = _build(tmp_path, jobs)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    result = subprocess.run([sys.executable, '-I', str(script)], capture_output=True, text=True, cwd=elsewhere)
    log = (tmp_path / 'runs' / 'job.log').read_text(encoding='utf-8')
    assert result.returncode == 2
    assert 'baseline_ncc exit 2' in log and 'start eval_position' not in log
    assert f'relative paths resolve from {tmp_path.resolve()}' in log
    assert '.psdl2_pkg_' in log and 'numpy ' in log


def test_etx_runner_is_current(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    try:
        import build_etx_runner
    finally:
        sys.path.pop(0)
    out = tmp_path / 'psdl2_job_baseline_ncc.py'
    build_etx_runner.main(['--out', str(out)])
    committed = ROOT / 'bundles/etx/psdl2_job_baseline_ncc.py'
    assert committed.read_bytes() == out.read_bytes(), 'rebuild with scripts/build_etx_runner.py --overwrite'
