"""Static review sheet for template-matching candidates (PLAN_v4 T2).

One PNG per query template and per top-K match window plus a single self-contained ``sheet.html``
(relative image files, no network, no scripts). Queries are listed by rank-1 score, highest first; the
score only orders and labels the page (S4), nothing is accepted or dropped here. Pixels are written
exactly as they are in the layout union: no resampling, the browser only magnifies them.
"""
import html
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .schema import ContractError

STYLE = '''body{font-family:sans-serif;margin:1.5em}table{border-collapse:collapse;margin-bottom:1.5em}
td,th{border:1px solid #999;padding:.3em .6em;vertical-align:top;font-size:.9em}
img{image-rendering:pixelated;min-width:96px;width:192px;border:1px solid #666;background:#000}
.flag{color:#a00;font-weight:bold}'''


def _png(mask):
    buffer = io.BytesIO()
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode='L').save(buffer, format='PNG')
    return buffer.getvalue()


def check_sheet_dir(sheet_dir):
    """Collision policy: the sheet is written into a directory that does not exist yet or is empty; anything already
    there (any name, any symlink) is refused, and no path component the caller can have planted may be a symlink
    (root-owned system links such as /tmp are tolerated). Returns the absolute directory path."""
    path = Path(os.path.abspath(sheet_dir))
    for part in (path, *path.parents):
        if part.is_symlink() and part.lstat().st_uid != 0:
            raise ContractError(f'sheet_dir_symlink_refused: {part}')
    if path.exists():
        if not path.is_dir():
            raise ContractError(f'sheet_dir_not_a_directory: {path}')
        if any(path.iterdir()):
            raise ContractError(f'sheet_dir_not_empty: {path}')
    return path


def _write_all(directory, files):
    """Create every file exclusively (O_EXCL|O_NOFOLLOW temp, then no-clobber hard link to the final name), so an
    existing file or link is never followed or replaced; on any failure whatever was created is removed again."""
    made = []
    try:
        for name, data in files.items():
            temp = directory / f'.{name}.tmp'
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o644)
            made.append(temp)
            with os.fdopen(fd, 'wb') as handle:
                handle.write(data)
            os.link(temp, directory / name)  # fails if the name exists
            made.append(directory / name)
            os.unlink(temp)
            made.remove(temp)
    except BaseException:
        for leftover in made:
            leftover.unlink(missing_ok=True)
        raise


def _fmt(value):
    return '-' if value is None else f'{value:.4f}'


def write_sheet(sheet_dir, entries, *, window_um, topk):
    """entries: dicts with roi_id, center_xy_um, status, reasons, score_gap, flags, shared_with, query_crop
    (2-D mask or None) and matches [{rank, score, center_xy_um, crop}]. Returns the sheet.html path."""
    sheet_dir = check_sheet_dir(sheet_dir)
    files = {}  # every planned output name → bytes, built before the first byte is written
    top = lambda e: e['matches'][0]['score'] if e['matches'] else None
    ordered = sorted(entries, key=lambda e: (top(e) is None, -(top(e) or 0.0), e['roi_id']))
    parts = ['<!DOCTYPE html>\n<html lang="ko"><head><meta charset="utf-8"><title>템플릿 매칭 검토 시트</title>',
             f'<style>{STYLE}</style></head><body>', '<h1>템플릿 매칭 검토 시트</h1>',
             f'<p>창 크기 {html.escape(str(window_um))} µm, 질의당 상위 {html.escape(str(topk))}개. '
             '점수는 정렬과 표시용이며 어떤 후보도 자동으로 채택하거나 버리지 않습니다. '
             '반복 형상에서는 점수가 거의 같은 후보가 여러 개 나올 수 있으니 이미지를 보고 사용자가 고르십시오.</p>']
    for index, entry in enumerate(ordered, 1):
        stem = f'q{index:04d}'
        flags = entry['flags']
        shown = [html.escape(name) for name, on in sorted(flags.items()) if on is True]
        parts.append(f'<h2>질의 {html.escape(entry["roi_id"])}</h2><table>')
        parts.append('<tr><th>중심 (µm)</th><th>순위 1 점수</th><th>점수 차 (1위−2위)</th><th>표시</th><th>같은 템플릿을 쓰는 질의</th></tr>')
        parts.append('<tr><td>{}</td><td>{}</td><td>{}</td><td class="flag">{}</td><td>{}</td></tr>'.format(
            html.escape(', '.join(f'{v:g}' for v in entry['center_xy_um'])), _fmt(top(entry)), _fmt(entry['score_gap']),
            ', '.join(shown + ([f'질의 창 무효 영역 {entry["query_invalid_pixels"]}px'] if entry['query_invalid_pixels'] else [])) or '&nbsp;', html.escape(', '.join(entry['shared_with'])) or '-'))
        parts.append('</table>')
        if entry['status'] != 'available':
            parts.append(f'<p>매칭 불가: {html.escape(", ".join(entry["reasons"]))}</p>')
            continue
        files[f'{stem}_query.png'] = _png(entry['query_crop'])
        parts.append(f'<table><tr><th>질의 템플릿</th>{"".join(f"<th>순위 {m["rank"]}</th>" for m in entry["matches"])}</tr><tr>')
        parts.append(f'<td><img src="{stem}_query.png" alt="질의 템플릿"></td>')
        for match in entry['matches']:
            name = f'{stem}_m{match["rank"]:02d}.png'
            files[name] = _png(match['crop'])
            parts.append(f'<td><img src="{name}" alt="순위 {match["rank"]} 후보"></td>')
        parts.append('</tr><tr><td>&nbsp;</td>')
        for match in entry['matches']:
            hole = f'<br><span class="flag">무효 영역 {match["invalid_pixels"]}px</span>' if match['invalid_pixels'] else ''
            parts.append('<td>점수 {}<br>중심 {}{}</td>'.format(
                _fmt(match['score']), html.escape(', '.join(f'{v:g}' for v in match['center_xy_um'])), hole))
        parts.append('</tr></table>')
    parts.append('</body></html>\n')
    files['sheet.html'] = '\n'.join(parts).encode('utf-8')
    sheet_dir.mkdir(parents=True, exist_ok=True)
    _write_all(sheet_dir, files)
    return sheet_dir / 'sheet.html'
