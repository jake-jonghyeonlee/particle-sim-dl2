#!/usr/bin/env python3
"""src/psdl2/*.py 전체를 .py 파일 하나로 묶는다. 명령줄 옵션을 넘길 수 없고 .py만 올릴 수 있는 서버(ETX)용.

만들어진 파일 맨 위의 EDIT 블록(JOBS, LOG_FILE, WORK_DIR, BASE_DIR)만 고치면 된다. 실행하면 패키지를 실제 폴더에 풀고(바이트 그대로)
psdl2.<모듈>.main(인자)을 순서대로 호출한다. baseline_ncc·eval_position은 numpy와 Pillow만 있으면 되고
(torch 불필요), train·search·evaluate 등은 torch가 필요하다. Python 3.8 이상."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = '''#!/usr/bin/env python3
"""psdl2(DL2) 단일 파일 실행기 (자동 생성: scripts/build_single_file.py). 소스 해시 @@SHA@@.
서버 wrapper가 명령줄 인자를 넘기지 못할 때 쓴다. 아래 EDIT 블록만 고치고 이 파일을 그대로 실행한다."""
# ============================== EDIT BELOW: 이 블록만 고치세요 ==============================
# 각 작업은 `python -m psdl2.<모듈> <인자...>`와 같다. 모듈 이름은 baseline_ncc, eval_position(이 둘은 torch 불필요),
# train, search, evaluate 등이다.
# 인자는 명령줄에서 쓰던 그대로 문자열 리스트로 적는다. 상대 경로는 BASE_DIR 기준이다(아래). wrapper가 이 파일을 다른 곳에
# 복사해 실행하면 "이 .py가 있는 폴더"가 달라지므로, 상대 경로를 쓸 때는 BASE_DIR를 절대 경로로 적는 것이 안전하다.
# 작업은 위에서부터 차례로 실행된다. 종료 코드 0(완료)과 3(부분 결과)은 다음 작업으로 넘어가고, 그 밖의 코드는 멈춘다.
JOBS = [
    ("baseline_ncc", ["--all-ready", "--allow-unversioned-producer",
                      "--query-export", "data/candidates_k3/reference_export/review_rois.json",
                      "--index-export", "data/candidates_k3/target_index_export/review_rois.json",
                      "--window-um", "20", "--topk", "4", "--max-memory-mb", "8000",
                      "--sheet-dir", "data/dl2/ncc_r2t_sheet", "--out", "data/dl2/ncc_r2t.json"]),
]
LOG_FILE = "runs/psdl2_job.log"               # stdout/stderr를 이 파일에도 남긴다. 필요 없으면 None
WORK_DIR = None                               # 패키지를 풀 폴더. None이면 이 파일 옆, 쓸 수 없으면 임시 폴더
BASE_DIR = None                               # JOBS·LOG_FILE·WORK_DIR의 상대 경로 기준 폴더(절대 경로). None이면 이 .py가 있는 폴더
# ============================== EDIT ABOVE ==============================================

import base64
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback
import zlib
from pathlib import Path

CONTINUE_CODES = (0, 3)
DONE_SUFFIX = ".psdl2_done.json"
# 완료 표지는 이 모듈에만 쓴다. 표지는 인자에 적힌 파일만 비교하므로, 간접 입력(예: 데이터셋이 가리키는 export, 체크포인트가
# 읽는 파일)이 있는 모듈은 표지를 쓰지도 읽지도 않고 매번 실행한다.
MARKER_MODULES = ("baseline_ncc", "eval_position")
TORCH_IMPORT = {"error": None}   # 버전 줄을 만들 때 첫 torch import 예외(형식, 메시지)를 저장한다
# 작업(모듈)별 바깥 의존. 표에 없는 모듈은 numpy·Pillow 만 쓴다(patches, data 포함).
# 이 표는 src/psdl2 의 실제 import 를 확인해 만든다: train·index·make_split·evaluate 는 import 때, search 는 실행 때 torch 를 쓴다.
MODULE_NEEDS = {
    "baseline_ncc": (), "eval_position": (), "patches": (), "data": (),
    "train": ("torch",), "index": ("torch",), "search": ("torch",), "evaluate": ("torch",), "make_split": ("torch",),
}
# 작업별 산출물 옵션. 표에 없으면 --out 하나다. 폴더 값이면 그 안의 모든 파일이 산출물이다.
OUTPUT_OPTIONS = {"baseline_ncc": ("--out", "--sheet-dir")}
PAYLOAD_SHA256 = "@@SHA@@"
PAYLOAD = (
@@PAYLOAD@@
)


def _files():
    files = json.loads(zlib.decompress(base64.b64decode("".join(PAYLOAD))))
    for name, entry in files.items():
        data = base64.b64decode(entry["data"])
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"embedded file is corrupted: {name}")
        files[name] = data
    return files


def _extract():
    """psdl2 패키지를 실제 폴더에 푼다(바이트 그대로). 이미 같은 내용이 있으면 재사용한다. 풀린 위치를 돌려준다."""
    files = _files()
    here = globals().get("__file__")
    bases = [Path(WORK_DIR)] if WORK_DIR else []
    bases += ([Path(here).resolve().parent] if here else [Path.cwd()]) + [Path(tempfile.gettempdir())]
    errors = []
    for base in bases:
        root = base / f".psdl2_pkg_{PAYLOAD_SHA256[:12]}"
        try:
            for name, data in files.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.is_file() or target.read_bytes() != data:
                    target.write_bytes(data)
            return root
        except OSError as exc:
            errors.append(f"{base}: {exc}")
    raise RuntimeError("cannot extract the psdl2 package anywhere: " + "; ".join(errors))


def _console_write(stream, text):
    """콘솔은 인코딩 오류를 '?'로 바꿔 흡수한다. 콘솔 때문에 작업이 죽지 않는다."""
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        stream.write(text.encode(encoding, "replace").decode(encoding, "replace"))


class _Tee:
    """로그 파일(UTF-8)에 먼저 쓰고 콘솔에 쓴다. 로그 파일이 없어도(LOG_FILE=None) 콘솔 쓰기는 같은 방식으로 흡수한다."""

    def __init__(self, log, console):
        self.log = log
        self.console = console

    def write(self, text):
        if self.log is not None:
            self.log.write(text)
        _console_write(self.console, text)
        return len(text)

    def flush(self):
        if self.log is not None:
            self.log.flush()
        self.console.flush()

    def isatty(self):
        return False


def _fail(code, english, korean, early=False):
    """오류 코드와 조치 문구를 stderr(와 로그)에 한 줄로 쓰고 종료 코드 2로 멈춘다. traceback 은 없다.
    ASCII 콘솔은 한글을 못 보여 주므로 영어 문구를 앞에 둔다. early=True 는 로그 파일을 열기 전 단계다."""
    line = f"[psdl2_job] ERROR {code}: {english} | {korean}\\n"
    if early:
        _console_write(sys.__stderr__, line)
    else:
        sys.stderr.write(line)
        sys.stderr.flush()
    raise SystemExit(2)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rel(path):
    """현재 작업 폴더(BASE_DIR) 아래면 상대 경로(/ 구분), 아니면 절대 경로."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _option_values(argv, options):
    """argv 에서 `--옵션 값`·`--옵션=값` 의 값을 모은다(옵션당 값 하나)."""
    values = []
    index = 0
    while index < len(argv):
        key, equals, value = argv[index].partition("=")
        if key in options:
            if equals:
                values.append(value)
            elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                values.append(argv[index + 1])
                index += 1
        index += 1
    return values


def _marker_path(name, argv):
    """표지 파일 경로. 표지 대상이 아닌 모듈이나 --out 이 없는 작업은 None."""
    if name not in MARKER_MODULES:
        return None
    out = _option_values(argv, ("--out",))
    if not out:
        return None
    if out[0].endswith(("/", "\\\\")):
        _fail("out_must_be_file_path", f"--out {out[0]!r} of job {name} ends with a path separator; give a file path, not a folder",
              f"{name} 작업의 --out {out[0]!r} 가 구분자로 끝난다. 폴더가 아니라 파일 경로를 적는다.")
    return Path(out[0] + DONE_SUFFIX)


def _output_files(name, argv):
    """작업이 만든(만들) 산출물 파일 목록. 폴더 값이면 안의 모든 파일."""
    files = []
    for value in _option_values(argv, OUTPUT_OPTIONS.get(name, ("--out",))):
        path = Path(value)
        if path.is_dir():
            files += [p for p in path.rglob("*") if p.is_file()]
        elif path.is_file():
            files.append(path)
    return sorted(files, key=lambda p: _rel(p))


def _input_hashes(name, argv):
    """argv 에 적힌 기존 파일 중 산출물이 아닌 것(입력)의 sha256. 키는 적힌 그대로의 경로."""
    outputs = set(_option_values(argv, OUTPUT_OPTIONS.get(name, ("--out",))))
    hashes = {}
    for token in argv:
        for candidate in (token, token.partition("=")[2]):
            if candidate and candidate not in outputs and Path(candidate).is_file():
                hashes[candidate] = _sha256_file(candidate)
    return hashes


def _marker_state(marker, name, argv):
    """None 이면 표지와 현재 상태가 모두 같다(건너뛴다). 아니면 (사유, 한글 설명, 영어 설명)."""
    try:
        done = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(done, dict):
            raise ValueError("not an object")
        if (not isinstance(done.get("module"), str) or not isinstance(done.get("runner_payload_sha256"), str)
                or not isinstance(done.get("argv"), list) or not isinstance(done.get("input_sha256s"), dict)
                or not isinstance(done.get("outputs"), list)
                or not all(isinstance(e, dict) and isinstance(e.get("path"), str) and isinstance(e.get("sha256"), str)
                           for e in done["outputs"])):
            raise ValueError("a key is missing or has the wrong type")
    except (OSError, ValueError) as exc:
        return "marker_unreadable", f"표지를 읽을 수 없다 ({type(exc).__name__})", "the marker cannot be read"
    if done.get("runner_payload_sha256") != PAYLOAD_SHA256:
        return "runner_changed", "표지를 만든 때와 이 실행기의 psdl2 판이 다르다", "this runner bundles a different psdl2 build than the one that wrote the marker"
    if done.get("module") != name or done.get("argv") != argv:
        return "inputs_changed", "표지를 만든 때와 작업 모듈 또는 인자가 다르다", "the job module or its arguments differ from the marker"
    recorded, current = done["input_sha256s"], _input_hashes(name, argv)
    if recorded != current:
        changed = sorted(k for k in set(recorded) | set(current) if recorded.get(k) != current.get(k))
        return "inputs_changed", "입력 파일이 표지를 만든 때와 다르다: " + ", ".join(changed), "input files differ from the marker: " + ", ".join(changed)
    for entry in done["outputs"]:
        path = Path(entry["path"])
        if not path.is_file():
            return "output_missing", f"산출물이 없다: {entry['path']}", f"output file is missing: {entry['path']}"
        if _sha256_file(path) != entry["sha256"]:
            return "output_modified", f"산출물이 표지 기록과 다르다: {entry['path']}", f"output file differs from the marker: {entry['path']}"
    return None


def _write_marker(marker, name, argv, inputs):
    outputs = _output_files(name, argv)
    if not any(Path(v).is_file() or Path(v).is_dir() for v in _option_values(argv, ("--out",))):
        _fail("job_output_missing", f"{name} exited 0 but its --out output is missing; no done marker is written",
              f"{name} 이 종료 코드 0 인데 --out 산출물이 없다. 완료 표지를 쓰지 않는다.")
    done = {"module": name, "runner_payload_sha256": PAYLOAD_SHA256, "argv": argv, "input_sha256s": inputs,
            "outputs": [{"path": _rel(p), "sha256": _sha256_file(p)} for p in outputs]}
    temp = marker.with_name(marker.name + f".tmp{os.getpid()}")
    temp.write_text(json.dumps(done, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    os.replace(str(temp), str(marker))


def _check_requirements(name):
    """작업 실행 전에 바깥 의존(torch)이 있는지 본다. 없으면 오류 코드로 멈춘다."""
    for dependency in MODULE_NEEDS.get(name, ()):
        if importlib.util.find_spec(dependency) is None:
            _fail(f"{dependency}_required_for_{name}",
                  f"job {name} needs {dependency}, which this Python ({sys.version.split()[0]}) does not have."
                  f" Run this file with a Python that has {dependency}, or remove the {name} job from JOBS."
                  " Outputs of earlier jobs are kept",
                  f"{name} 작업에는 {dependency} 가 필요한데 이 Python({sys.version.split()[0]})에 없다."
                  f" {dependency} 가 설치된 Python 으로 이 파일을 실행하거나, JOBS 에서 {name} 작업을 뺀다."
                  " 앞 작업의 산출물은 그대로 남아 있다.")
        # torch 는 버전 줄을 만들 때 이미 한 번 import 를 시도했다. 다시 import 하지 않고 그 예외를 쓴다.
        error = TORCH_IMPORT["error"] if dependency == "torch" else None
        if error is None and dependency != "torch":
            try:
                importlib.import_module(dependency)
            except Exception as exc:
                error = (type(exc).__name__, str(exc))
        if error is not None:
            _fail(f"{dependency}_import_failed",
                  f"{error[0]}: {error[1]}. Importing {dependency} for job {name} failed."
                  f" Check the {dependency} installation (CUDA and shared libraries). Outputs of earlier jobs are kept",
                  f"{name} 작업을 위한 {dependency} import 가 실패했다."
                  f" {dependency} 설치(CUDA·라이브러리 포함)를 확인한다. 앞 작업의 산출물은 그대로 남아 있다.")


@@HOOKS@@


def run_jobs(jobs):
    root = _extract()
    sys.path.insert(0, str(root))
    import psdl2
    if not str(Path(psdl2.__file__).resolve()).startswith(str(root.resolve())):
        raise RuntimeError(f"another psdl2 shadows the bundled one: {psdl2.__file__}")
    stamp = lambda: time.strftime("%Y-%m-%d %H:%M:%S")
    import numpy
    versions = [f"python {sys.version.split()[0]}", f"numpy {numpy.__version__}"]
    for optional in ("PIL", "torch"):
        try:
            module = importlib.import_module(optional)
            versions.append(f"{optional} {module.__version__}")
            if optional == "torch":
                versions.append(f"cuda available: {bool(module.cuda.is_available())}")
        except Exception as exc:  # 로그용 정보일 뿐이다. torch 예외는 작업 전 검사가 다시 import 하지 않고 쓴다.
            versions.append(f"{optional} unavailable ({type(exc).__name__})")
            if optional == "torch" and not (isinstance(exc, ModuleNotFoundError) and exc.name == "torch"):
                TORCH_IMPORT["error"] = (type(exc).__name__, str(exc))
    print(f"[psdl2_job] {stamp()} {' | '.join(versions)} | psdl2 from {psdl2.__file__}", file=sys.stderr, flush=True)
    _prepare(Path.cwd())
    for name, args in jobs:
        argv = [str(a) for a in args]
        marker = _marker_path(name, argv)
        if name not in MARKER_MODULES:
            print(f"[psdl2_job] {stamp()} no done marker for {name} (indirect inputs are not hashed)", file=sys.stderr, flush=True)
        if marker is not None and marker.exists():
            stale = _marker_state(marker, name, argv)
            if stale is None:
                print(f"[psdl2_job] {stamp()} skip completed {name} (done marker {_rel(marker)})", file=sys.stderr, flush=True)
                continue
            _fail(f"stale_done_marker:{stale[0]}",
                  f"done marker {_rel(marker)} does not match job {name} ({stale[2]})."
                  " Delete the marker and run again (if the job needs an empty output folder, move the old outputs away too)",
                  f"{stale[1]}. 완료 표지 {_rel(marker)} 가 이 작업과 맞지 않는다."
                  " 표지를 지우고 다시 실행한다(산출물 폴더가 비어 있어야 하는 작업이면 옛 산출물도 옮긴다).")
        _check_requirements(name)
        inputs = _input_hashes(name, argv) if marker is not None else None
        print(f"[psdl2_job] {stamp()} start {name} {' '.join(argv)}", file=sys.stderr, flush=True)
        started = time.time()
        main = importlib.import_module(f"psdl2.{name}").main
        sys.argv = [f"psdl2.{name}", *argv]
        try:
            code = main(argv)
        except SystemExit as exc:  # argparse 오류 등
            code = exc.code if isinstance(exc.code, int) else 2
        code = 0 if code is None else code
        print(f"[psdl2_job] {stamp()} {name} exit {code} in {time.time() - started:.0f}s", file=sys.stderr, flush=True)
        if code not in CONTINUE_CODES:
            raise SystemExit(code)
        if code == 0 and marker is not None:
            _write_marker(marker, name, argv, inputs)
            print(f"[psdl2_job] {stamp()} wrote done marker {_rel(marker)}", file=sys.stderr, flush=True)
    _check(Path.cwd())


def _base_dir():
    """상대 경로 기준 폴더: BASE_DIR, 없으면 이 파일의 폴더, __file__도 없으면 현재 작업 폴더."""
    if BASE_DIR:
        return Path(BASE_DIR).expanduser().resolve(), "BASE_DIR"
    here = globals().get("__file__")
    if here:
        return Path(here).resolve().parent, "runner file folder"
    return Path.cwd(), "current working directory (__file__ unavailable)"


def _main():
    base, source = _base_dir()
    launched_cwd = os.getcwd()
    if not base.is_dir():
        _fail("base_dir_missing", f"the folder {base} ({source}) does not exist."
                                  " Set BASE_DIR to the absolute path of an existing folder",
              f"{source} 가 가리키는 폴더 {base} 가 없다. BASE_DIR 를 존재하는 폴더의 절대 경로로 고친다.", early=True)
    os.chdir(base)
    log = None
    if LOG_FILE:
        try:
            Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
            log = open(LOG_FILE, "a", buffering=1, encoding="utf-8")
        except OSError as exc:
            _fail("log_file_unwritable", f"cannot write LOG_FILE {LOG_FILE!r} under {base} ({type(exc).__name__}: {exc})."
                                         " Set LOG_FILE to a writable path, or to None",
                  f"LOG_FILE {LOG_FILE!r} 를 {base} 아래에 쓸 수 없다. LOG_FILE 을 쓸 수 있는 경로로 고치거나 None 으로 둔다.", early=True)
    sys.stdout, sys.stderr = _Tee(log, sys.__stdout__), _Tee(log, sys.__stderr__)
    print(f"[psdl2_job] relative paths resolve from {base} ({source}); launched in {launched_cwd};"
          f" runner {globals().get('__file__')}", file=sys.stderr, flush=True)
    try:
        run_jobs(JOBS)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if log is not None:
            log.flush()


if __name__ == "__main__":
    _main()
'''


def _sources():
    """묶을 파일: src/psdl2 아래 모든 .py."""
    package = ROOT / "src"
    return {p.relative_to(package).as_posix(): p for p in sorted((package / "psdl2").glob("*.py"))}


def _jobs_block(jobs):
    """JOBS 리스트를 EDIT 블록에 넣을 파이썬 소스로 만든다. jobs는 (명령, [인자...], 주석 또는 None) 목록이다."""
    lines = ["JOBS = ["]
    for name, args, comment in jobs:
        if comment:
            lines.append(f"    # {comment}")
        lines.append(f"    ({name!r}, [")
        index = 0
        while index < len(args):
            # "--옵션 값"은 한 줄에 둔다.
            pair = (args[index].startswith("--") and index + 1 < len(args) and not args[index + 1].startswith("--"))
            chunk = args[index:index + 2] if pair else args[index:index + 1]
            lines.append("        " + " ".join(f"{a!r}," for a in chunk))
            index += len(chunk)
        lines.append("    ]),")
    lines.append("]")
    return "\n".join(lines) + "\n"


DEFAULT_HOOKS = '''def _prepare(base):
    """작업을 시작하기 전에 한 번 부르는 준비 단계. 기본은 아무것도 하지 않는다."""


def _check(base):
    """모든 작업이 끝난 뒤 한 번 부르는 확인 단계. 기본은 아무것도 하지 않는다."""
'''


def build(out, *, overwrite=False, jobs=None, log_file=None, header=None, jobs_source=None, extra_edit=None, hooks=None,
          docstring=None):
    """jobs_source: JOBS 블록을 이 소스 텍스트로 바꾼다(jobs 대신). extra_edit: EDIT 블록 맨 앞에 넣을 소스(예: DEVICE).
    hooks: _prepare(base)·_check(base) 를 정의하는 소스. docstring: 실행기 맨 위 설명."""
    out = Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {out}; pass --overwrite or choose a new path")
    sources = _sources()
    if not any(name.startswith("psdl2/") for name in sources):
        raise ValueError("no source files found under src/psdl2")
    files = {name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "data": base64.b64encode(p.read_bytes()).decode()}
             for name, p in sources.items()}
    packed = base64.b64encode(zlib.compress(json.dumps(files, sort_keys=True).encode(), 9)).decode()
    sha = hashlib.sha256(json.dumps({n: e["sha256"] for n, e in files.items()}, sort_keys=True).encode()).hexdigest()
    lines = "\n".join(f'    "{packed[i:i + 100]}"' for i in range(0, len(packed), 100))
    text = TEMPLATE.replace("@@SHA@@", sha).replace("@@PAYLOAD@@", lines)
    if jobs is not None or jobs_source is not None:
        start, end = text.index("JOBS = ["), text.index("LOG_FILE = ")
        text = text[:start] + (extra_edit or "") + (jobs_source if jobs_source is not None else _jobs_block(jobs)) + text[end:]
    if docstring is not None:
        start = text.index('"""psdl2(DL2)')
        end = text.index('"""', start + 3) + 3
        text = text[:start] + '"""' + docstring.replace("@@SHA@@", sha) + '"""' + text[end:]
    text = text.replace("@@HOOKS@@", (hooks or DEFAULT_HOOKS).rstrip("\n"))
    if log_file is not None:
        start = text.index("LOG_FILE = ")
        end = text.index("\n", start)
        text = text[:start] + f"LOG_FILE = {log_file!r}   # stdout/stderr를 이 파일에도 남긴다. 필요 없으면 None" + text[end:]
    if header:
        marker = "# ============================== EDIT BELOW"
        text = text.replace(marker, "".join(f"# {line}\n" for line in header) + marker, 1)
    compile(text, str(out), "exec")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return {"out": str(out), "bytes": out.stat().st_size, "modules": len(files),
            "source_sha256": sha}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="single .py file to write, e.g. bundles/psdl2_job.py")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.out, overwrite=a.overwrite), indent=2))


if __name__ == "__main__":
    main()
