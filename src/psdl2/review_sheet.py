"""Static review sheet for template-matching candidates (PLAN_v4 T2).

One PNG per query template and per top-K match window plus a single self-contained ``sheet.html``
(relative image files, no network, no scripts). Queries are listed by rank-1 score, highest first; the
score only orders and labels the page (S4), nothing is accepted or dropped here. Pixels are written
exactly as they are in the layout union: no resampling, the browser only magnifies them.
"""
import html
from pathlib import Path

import numpy as np
from PIL import Image

STYLE = '''body{font-family:sans-serif;margin:1.5em}table{border-collapse:collapse;margin-bottom:1.5em}
td,th{border:1px solid #999;padding:.3em .6em;vertical-align:top;font-size:.9em}
img{image-rendering:pixelated;min-width:96px;width:192px;border:1px solid #666;background:#000}
.flag{color:#a00;font-weight:bold}'''


def _png(path, mask):
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode='L').save(path)


def _fmt(value):
    return '-' if value is None else f'{value:.4f}'


def write_sheet(sheet_dir, entries, *, window_um, topk):
    """entries: dicts with roi_id, center_xy_um, status, reasons, score_gap, flags, shared_with, query_crop
    (2-D mask or None) and matches [{rank, score, center_xy_um, crop}]. Returns the sheet.html path."""
    sheet_dir = Path(sheet_dir)
    sheet_dir.mkdir(parents=True, exist_ok=True)
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
            ', '.join(shown) or '&nbsp;', html.escape(', '.join(entry['shared_with'])) or '-'))
        parts.append('</table>')
        if entry['status'] != 'available':
            parts.append(f'<p>매칭 불가: {html.escape(", ".join(entry["reasons"]))}</p>')
            continue
        _png(sheet_dir / f'{stem}_query.png', entry['query_crop'])
        parts.append(f'<table><tr><th>질의 템플릿</th>{"".join(f"<th>순위 {m["rank"]}</th>" for m in entry["matches"])}</tr><tr>')
        parts.append(f'<td><img src="{stem}_query.png" alt="질의 템플릿"></td>')
        for match in entry['matches']:
            name = f'{stem}_m{match["rank"]:02d}.png'
            _png(sheet_dir / name, match['crop'])
            parts.append(f'<td><img src="{name}" alt="순위 {match["rank"]} 후보"></td>')
        parts.append('</tr><tr><td>&nbsp;</td>')
        for match in entry['matches']:
            parts.append('<td>점수 {}<br>중심 {}</td>'.format(
                _fmt(match['score']), html.escape(', '.join(f'{v:g}' for v in match['center_xy_um']))))
        parts.append('</tr></table>')
    parts.append('</body></html>\n')
    path = sheet_dir / 'sheet.html'
    path.write_text('\n'.join(parts), encoding='utf-8')
    return path
