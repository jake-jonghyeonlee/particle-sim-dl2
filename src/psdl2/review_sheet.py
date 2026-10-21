"""Static review sheet for template-matching candidates (PLAN_v4 T2).

One PNG per query template and per top-K match window plus a single self-contained ``sheet.html``
(relative image files, no network, no scripts). Queries are listed by rank-1 score, highest first; the
score only orders and labels the page (S4), nothing is accepted or dropped here. Pixels are written
exactly as they are in the layout union: no resampling, the browser only magnifies them.
"""
from __future__ import annotations

import html
import io
import os
from pathlib import Path
import stat

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


def _trusted_link(parent_fd, link_stat):
    """A symlink is followed only when nobody but root can have planted or can change it: the link and the directory
    holding it are root-owned and that directory is not group/world-writable (the macOS ``/tmp -> private/tmp`` case)."""
    parent = os.fstat(parent_fd)
    return (link_stat.st_uid == 0 and parent.st_uid == 0 and not parent.st_mode & (stat.S_IWGRP | stat.S_IWOTH))


def _open_dir(sheet_dir, *, create):
    """(absolute path, fd of the directory, or None when it does not exist and ``create`` is false).

    The path is walked once, component by component, with every directory opened O_DIRECTORY|O_NOFOLLOW relative to
    its parent's descriptor, so the result cannot have been redirected by a symlink and the returned descriptor
    stays bound to that directory whatever happens to the path afterwards. Missing components are created when
    ``create``. Symlinks are refused except trusted root-owned system links (see ``_trusted_link``), which are
    followed through the same descriptor walk; their targets must not contain ``..``."""
    path = os.path.abspath(sheet_dir)
    pending = [part for part in path.split('/') if part]
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | os.O_NOFOLLOW
    fd = os.open('/', flags)
    walked, followed = '', 0
    try:
        while pending:
            name = pending.pop(0)
            walked = f'{walked}/{name}'
            try:
                child = os.open(name, flags, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    os.close(fd)
                    return path, None
                try:
                    os.mkdir(name, 0o777, dir_fd=fd)
                except FileExistsError:
                    pass
                child = os.open(name, flags, dir_fd=fd)
            except OSError:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if not stat.S_ISLNK(info.st_mode):
                    raise ContractError(f'sheet_dir_not_a_directory: {walked}') from None
                followed += 1
                if followed > 8 or not _trusted_link(fd, info):
                    raise ContractError(f'sheet_dir_symlink_refused: {walked}') from None
                target = os.readlink(name, dir_fd=fd)
                parts = [part for part in target.split('/') if part and part != '.']
                if '..' in parts:
                    raise ContractError(f'sheet_dir_symlink_refused: {walked} -> {target}') from None
                if target.startswith('/'):
                    os.close(fd)
                    fd, walked = os.open('/', flags), ''
                else:
                    walked = walked.rsplit('/', 1)[0]
                pending[:0] = parts
                continue
            os.close(fd)
            fd = child
        return path, fd
    except BaseException:
        os.close(fd)
        raise


def _require_empty(fd, path):
    if os.listdir(fd):
        raise ContractError(f'sheet_dir_not_empty: {path}')


def check_sheet_dir(sheet_dir):
    """Collision policy: the sheet is written into a directory that does not exist yet or is empty; anything already
    there (any name, any symlink) is refused, and no path component the caller can have planted may be a symlink
    (root-owned system links such as /tmp are followed, see ``_trusted_link``). Returns the absolute directory path."""
    path, fd = _open_dir(sheet_dir, create=False)
    if fd is not None:
        try:
            _require_empty(fd, path)
        finally:
            os.close(fd)
    return Path(path)


def _same_inode(a, b):
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _write_all(dir_fd, files, verify=None):
    """Create every file in the directory ``dir_fd`` is bound to, exclusively (O_EXCL|O_NOFOLLOW temp, then no-clobber
    hard link to the final name), so an existing file or link is never followed or replaced. Every operation is relative
    to ``dir_fd``; each created entry is remembered by (st_dev, st_ino). On any failure, or when ``verify()`` raises after
    everything is published, exactly the entries this call created are unlinked again (identity re-checked), never a
    different file that now carries the same name."""
    made = {}  # name → (st_dev, st_ino) of what this call created

    def remove(name):
        try:
            current = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) == made[name]:
            os.unlink(name, dir_fd=dir_fd)

    try:
        for name, data in files.items():
            temp = f'.{name}.tmp'
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dir_fd)
            with os.fdopen(fd, 'wb') as handle:
                info = os.fstat(handle.fileno())
                made[temp] = (info.st_dev, info.st_ino)
                handle.write(data)
            os.link(temp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd, follow_symlinks=False)  # fails if the name exists
            made[name] = made[temp]
            remove(temp)
            del made[temp]
        if verify is not None:
            verify()
    except BaseException:
        for name in list(made):
            try:
                remove(name)
            except OSError:
                pass
        raise


def _fmt(value):
    return '-' if value is None else f'{value:.4f}'


def write_sheet(sheet_dir, entries, *, window_um, topk):
    """entries: dicts with roi_id, center_xy_um, status, reasons, score_gap, flags, shared_with, query_crop
    (2-D mask or None) and matches [{rank, score, center_xy_um, crop}]. Returns the sheet.html path."""
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
        rank_headers = ''.join('<th>순위 {}</th>'.format(m['rank']) for m in entry['matches'])  # 3.12 이전 f-string 호환
        parts.append(f'<table><tr><th>질의 템플릿</th>{rank_headers}</tr><tr>')
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
    path, dir_fd = _open_dir(sheet_dir, create=True)
    try:
        _require_empty(dir_fd, path)
        held = os.fstat(dir_fd)

        def still_there():  # the directory we wrote into must still be what the path names (no rename/replacement meanwhile)
            _, fd = _open_dir(path, create=False)
            try:
                if fd is None or not _same_inode(os.fstat(fd), held):
                    raise ContractError(f'sheet_dir_replaced_during_write: {path}')
            finally:
                if fd is not None:
                    os.close(fd)

        _write_all(dir_fd, files, verify=still_there)
    finally:
        os.close(dir_fd)
    return Path(path) / 'sheet.html'
