#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Optional


def infer_repo_root(assembled_path: Path) -> Path:
    start = assembled_path.resolve()
    for cand in [start.parent] + list(start.parents):
        if (cand / 'process').exists() and (cand / 'input').exists():
            return cand
        if cand.name.lower() == 'process' and (cand.parent / 'input').exists():
            return cand.parent
    return assembled_path.parent.parent.parent


def magic_type(p: Path) -> str:
    try:
        b = p.read_bytes()[:16]
    except Exception:
        return 'unknown'
    if len(b) >= 3 and b[0:3] == b'\xFF\xD8\xFF':
        return 'jpeg'
    if len(b) >= 8 and b[0:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    if len(b) >= 12 and b[0:4] == b'RIFF' and b[8:12] == b'WEBP':
        return 'webp'
    if len(b) >= 6 and (b[0:6] == b'GIF87a' or b[0:6] == b'GIF89a'):
        return 'gif'
    return 'unknown'


def iter_image_paths(assembled: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    for s in (assembled.get('slides') or []):
        imgs = s.get('images') or []
        for im in imgs:
            if isinstance(im, str):
                paths.append(im)
            elif isinstance(im, dict):
                p = im.get('path')
                if p:
                    paths.append(p)
    seen=set(); out=[]
    for p in paths:
        if p in seen:
            continue
        seen.add(p); out.append(p)
    return out


def search_under(repo_root: Path, name: str) -> Optional[Path]:
    for sub in ('process', 'input'):
        base = repo_root / sub
        if not base.exists():
            continue
        try:
            for hit in base.rglob(name):
                if hit.is_file():
                    return hit
        except Exception:
            pass
    return None


def resolve(img: str, base_dir: Path, repo_root: Path) -> Optional[Path]:
    p = Path(img)
    if p.is_absolute() and p.exists():
        return p
    cand = (base_dir / img).resolve()
    if cand.exists():
        return cand
    bn = p.name
    cand2 = (base_dir / bn).resolve()
    if cand2.exists():
        return cand2
    for r in (
        repo_root / 'process' / 'onenote',
        repo_root / 'input' / 'onenote-exporter',
        repo_root / 'input' / 'onenote-exporter' / 'output',
        repo_root / 'input',
    ):
        if not r.exists():
            continue
        c3 = r / img
        if c3.exists():
            return c3
        c4 = r / bn
        if c4.exists():
            return c4
    return search_under(repo_root, bn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assembled', required=True)
    ap.add_argument('--project-root', default='')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    assembled_path = Path(args.assembled)
    assembled = json.loads(assembled_path.read_text(encoding='utf-8'))
    base_dir = assembled_path.parent
    repo_root = Path(args.project_root) if args.project_root else infer_repo_root(assembled_path)

    imgs = iter_image_paths(assembled)
    results = []
    missing = 0
    mismatch = 0
    for img in imgs:
        ip = resolve(img, base_dir, repo_root)
        if not ip:
            missing += 1
            results.append({'requested': img, 'resolved': None, 'status': 'not_found'})
            continue
        kind = magic_type(ip)
        ext = ip.suffix.lower()
        ok = (kind == 'jpeg' and ext in ('.jpg','.jpeg')) or (kind == 'png' and ext == '.png')
        status = 'ok' if ok else 'format_mismatch'
        if status == 'format_mismatch':
            mismatch += 1
        results.append({'requested': img, 'resolved': str(ip), 'status': status, 'magic': kind, 'ext': ext})

    out_path = Path(args.out) if args.out else (assembled_path.with_suffix('.image_diagnostics.json'))
    payload = {
        'assembled': str(assembled_path),
        'repo_root': str(repo_root),
        'count_requested': len(imgs),
        'count_not_found': missing,
        'count_format_mismatch': mismatch,
        'samples_not_found': [r['requested'] for r in results if r['status']=='not_found'][:20],
        'samples_mismatch': [r for r in results if r['status']=='format_mismatch'][:20],
        'items': results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Repo root:', repo_root)
    print('Wrote:', out_path)
    print(f"Images: requested={len(imgs)} not_found={missing} format_mismatch={mismatch}")


if __name__ == '__main__':
    main()
