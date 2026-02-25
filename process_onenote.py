#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
process_onenote.py

Purpose
-------
Convert OneNote Markdown exports into structured, LLM-ready JSON "page packs".

Input
-----
onenote-exporter/output/<notebook>/**/*.md

Output
------
processed/<notebook>/
  pages/*.json
  assets/
  manifest.json
  errors.jsonl

Usage Example
--------------
python process_onenote.py test --transcribe
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# 1) Regex patterns & constants
# ============================================================

RE_FRONTMATTER = re.compile(r"^\s*---\s*$")
RE_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
RE_MD_IMAGE = re.compile(r"!\[\]\(([^)]+)\)")
RE_IMAGE_OCR = re.compile(r"\[\[IMAGE OCR\]\]\s*(.*)$")
RE_AUDIO_REC = re.compile(r"\[\[AUDIO RECORDING\]\]\s*\[Play\]\(([^)]+)\)", re.IGNORECASE)

AUDIO_EXTS = {".3gp", ".m4a", ".mp3", ".wav", ".aac", ".ogg", ".mp4"}


# ============================================================
# 2) Small utilities
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def normalize_line(line: str) -> str:
    return line.rstrip("\n\r ")


# ============================================================
# 3) Frontmatter parsing
# ============================================================

def parse_frontmatter_and_body(text: str) -> tuple[dict, str]:
    """
    Parse a simple YAML-like frontmatter delimited by '---'.
    """
    lines = text.splitlines()
    if not lines or not RE_FRONTMATTER.match(lines[0]):
        return {}, text

    end_idx = None
    for i in range(1, len(lines)):
        if RE_FRONTMATTER.match(lines[i]):
            end_idx = i
            break

    if end_idx is None:
        return {}, text

    meta = {}
    for raw in lines[1:end_idx]:
        if ":" in raw:
            k, v = raw.split(":", 1)
            meta[k.strip()] = v.strip()

    body = "\n".join(lines[end_idx + 1:])
    return meta, body


# ============================================================
# 4) Block builder
# ============================================================

def new_block(blocks: list, block_type: str, **kwargs) -> dict:
    block_id = f"b{len(blocks) + 1:05d}"
    blk = {"block_id": block_id, "type": block_type}
    blk.update(kwargs)
    blocks.append(blk)
    return blk


# ============================================================
# 5) Asset resolution
# ============================================================

def resolve_asset_path(md_file: Path, asset_ref: str, notebook_root: Path) -> Path:
    """
    Resolve an asset reference relative to notebook root.
    Falls back to filename search.
    """
    candidate = (notebook_root / asset_ref).resolve()
    if candidate.exists():
        return candidate

    fname = Path(asset_ref).name
    hits = list(notebook_root.rglob(fname))
    if hits:
        return hits[0].resolve()

    return candidate


# ============================================================
# 6) Audio transcription
# ============================================================

def transcribe_audio(asset_path: Path, lang: str = "fr") -> tuple[str | None, dict]:
    """
    Attempt transcription using:
      - openai-whisper
      - faster-whisper
    """
    meta = {
        "engine": None,
        "language": lang,
        "ffmpeg": which("ffmpeg"),
        "status": "not_run",
        "error": None,
    }

    if not meta["ffmpeg"]:
        meta["status"] = "failed"
        meta["error"] = "ffmpeg not found"
        return None, meta

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "audio.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(asset_path), "-ac", "1", "-ar", "16000", str(wav_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except Exception as e:
            meta["status"] = "failed"
            meta["error"] = f"ffmpeg conversion failed: {e}"
            return None, meta

        # Try openai-whisper
        try:
            import whisper
            meta["engine"] = "openai-whisper"
            model = whisper.load_model("small")
            result = model.transcribe(str(wav_path), language=lang)
            meta["status"] = "ok"
            return (result.get("text") or "").strip(), meta
        except Exception:
            pass

        # Try faster-whisper
        try:
            from faster_whisper import WhisperModel
            meta["engine"] = "faster-whisper"
            model = WhisperModel("small", device="cpu", compute_type="int8")
            segments, info = model.transcribe(str(wav_path), language=lang)
            text = " ".join(seg.text.strip() for seg in segments)
            meta["status"] = "ok"
            meta["info"] = {"language_probability": getattr(info, "language_probability", None)}
            return text.strip(), meta
        except Exception as e:
            meta["status"] = "failed"
            meta["error"] = f"No whisper engine available: {e}"
            return None, meta


# ============================================================
# 7) Markdown parsing
# ============================================================

def parse_markdown(md_file: Path,
                   meta: dict,
                   body: str,
                   transcribe: bool,
                   notebook_root: Path) -> dict:
    """
    Convert Markdown body into structured blocks.
    """
    blocks = []
    assets = {"images": [], "audio": []}
    errors = []
    last_image_id = None

    for idx, raw in enumerate(body.splitlines(), start=1):
        line = normalize_line(raw)
        if not line.strip():
            continue

        # Headings
        m = RE_MD_HEADING.match(line)
        if m:
            new_block(
                blocks,
                "heading",
                level=len(m.group(1)),
                text=m.group(2).strip(),
                source_line=idx,
            )
            continue

        # Audio
        m = RE_AUDIO_REC.search(line)
        if m:
            ref = m.group(1).strip()
            path = resolve_asset_path(md_file, ref, notebook_root)
            assets["audio"].append(ref)
            blk = new_block(
                blocks,
                "audio",
                path=ref,
                transcript=None,
                transcript_meta=None,
                source_line=idx,
            )
            if transcribe:
                if path.exists():
                    txt, tmeta = transcribe_audio(path)
                    blk["transcript"] = txt
                    blk["transcript_meta"] = tmeta
                    if txt is None:
                        errors.append({"file": str(md_file), "line": idx, "kind": "audio_transcription", "detail": tmeta.get("error")})
                else:
                    err = f"Audio not found: {path}"
                    blk["transcript_meta"] = {"status": "failed", "error": err}
                    errors.append({"file": str(md_file), "line": idx, "kind": "audio_missing", "detail": err})
            continue

        # Images
        m = RE_MD_IMAGE.search(line)
        if m:
            ref = m.group(1).strip()
            path = resolve_asset_path(md_file, ref, notebook_root)
            assets["images"].append(ref)
            img_blk = new_block(
                blocks,
                "image",
                path=ref,
                exists=path.exists(),
                source_line=idx,
            )
            last_image_id = img_blk["block_id"]
            continue

        # Image OCR
        m = RE_IMAGE_OCR.search(line)
        if m and last_image_id:
            new_block(
                blocks,
                "image_ocr",
                image_block_id=last_image_id,
                text=(m.group(1) or "").strip(),
                source_line=idx,
            )
            continue

        # Paragraph
        new_block(blocks, "paragraph", text=line.strip(), source_line=idx)

    return {
        "metadata": meta,
        "page_id": meta.get("page_id") or md_file.stem,
        "title": meta.get("title") or md_file.stem,
        "notebook": meta.get("notebook"),
        "section": meta.get("section"),
        "source_file": str(md_file),
        "generated_at": utc_now_iso(),
        "blocks": blocks,
        "assets": assets,
        "errors": errors,
    }


# ============================================================
# 8) Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Process OneNote markdown exports into JSON page packs.")
    ap.add_argument("notebook", help="Notebook name (matches frontmatter notebook:)")
    ap.add_argument("--input", default="onenote-exporter/output", help="Input root")
    ap.add_argument("--out", default="processed", help="Output root")
    ap.add_argument("--transcribe", action="store_true", help="Enable audio transcription")
    ap.add_argument("--copy-assets", action="store_true", help="Copy image/audio assets")
    args = ap.parse_args()

    input_root = Path(args.input).resolve()
    notebook_root = input_root / args.notebook
    out_root = Path(args.out).resolve() / args.notebook
    pages_dir = out_root / "pages"
    assets_dir = out_root / "assets"

    pages_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_assets:
        (assets_dir / "images").mkdir(parents=True, exist_ok=True)
        (assets_dir / "audio").mkdir(parents=True, exist_ok=True)

    manifest = {
        "notebook": args.notebook,
        "generated_at": utc_now_iso(),
        "processed_pages": [],
        "skipped_files": [],
        "errors": [],
    }

    for md_file in input_root.rglob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
            meta, body = parse_frontmatter_and_body(text)

            if meta.get("notebook") != args.notebook:
                manifest["skipped_files"].append(str(md_file))
                continue

            page = parse_markdown(md_file, meta, body, args.transcribe, notebook_root)
            out_path = pages_dir / f"{page['page_id'].replace('/', '_')}.json"
            out_path.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest["processed_pages"].append(page["page_id"])

        except Exception as e:
            manifest["errors"].append({"file": str(md_file), "error": str(e)})

    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()