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

RE_FRONTMATTER = re.compile(r"^\s*---\s*$")
RE_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
RE_MD_IMAGE = re.compile(r"!\[\]\(([^)]+)\)")

RE_IMAGE_OCR = re.compile(r"(?:\\\\\[|\[)IMAGE OCR(?:\\\\\]|\])\s*(.*)$")
RE_AUDIO_REC = re.compile(r"(?:\\\\\[|\[)AUDIO RECORDING(?:\\\\\]|\])\s*\[Play\]\(([^)]+)\)", re.IGNORECASE)

AUDIO_EXTS = {".3gp", ".m4a", ".mp3", ".wav", ".aac", ".ogg", ".mp4"}


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def parse_frontmatter_and_body(text: str):
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

    fm_lines = lines[1:end_idx]
    body_lines = lines[end_idx + 1:]
    meta = {}
    for raw in fm_lines:
        if ":" in raw:
            k, v = raw.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, "\n".join(body_lines)


def normalize_line(line: str) -> str:
    return line.rstrip("\n\r ")


def new_block(blocks, btype, **kwargs):
    block_id = f"b{len(blocks)+1:05d}"
    blk = {"block_id": block_id, "type": btype}
    blk.update(kwargs)
    blocks.append(blk)
    return blk



def resolve_asset_path(md_file: Path, asset_ref: str, notebook_root: Path) -> Path:
    base = notebook_root

    candidate = (base / asset_ref).resolve()
    if candidate.exists():
        return candidate

    fname = Path(asset_ref).name
    hits = list(base.rglob(fname))
    if hits:
        return hits[0].resolve()

    return candidate


def transcribe_audio(asset_path: Path, lang: str = "fr"):
    meta = {
        "engine": None,
        "language": lang,
        "ffmpeg": which("ffmpeg"),
        "status": "not_run",
        "error": None,
    }

    if not meta["ffmpeg"]:
        meta["status"] = "failed"
        meta["error"] = "ffmpeg not found in PATH"
        return None, meta

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "audio.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(asset_path),
            "-ac", "1",
            "-ar", "16000",
            str(wav_path)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as e:
            meta["status"] = "failed"
            meta["error"] = f"ffmpeg conversion failed: {e}"
            return None, meta

        try:
            import whisper
            meta["engine"] = "openai-whisper"
            meta["status"] = "running"
            model = whisper.load_model("small")
            result = model.transcribe(str(wav_path), language=lang)
            meta["status"] = "ok"
            return (result.get("text") or "").strip(), meta
        except Exception:
            pass

        try:
            from faster_whisper import WhisperModel
            meta["engine"] = "faster-whisper"
            meta["status"] = "running"
            model = WhisperModel("small", device="cpu", compute_type="int8")
            segments, info = model.transcribe(str(wav_path), language=lang)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            meta["status"] = "ok"
            meta["info"] = {"language_probability": getattr(info, "language_probability", None)}
            return text, meta
        except Exception as e:
            meta["status"] = "failed"
            meta["error"] = "No whisper engine available (install openai-whisper or faster-whisper). " + str(e)
            return None, meta


def parse_markdown(md_file: Path, meta: dict, body: str, transcribe: bool, notebook_root: Path):
    blocks = []
    assets = {"images": [], "audio": []}
    errors = []

    last_image_id = None

    for idx, raw in enumerate(body.splitlines(), start=1):
        line = normalize_line(raw)
        if not line.strip():
            continue

        m = RE_MD_HEADING.match(line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            new_block(blocks, "heading", level=level, text=text, source_line=idx)
            continue

        m = RE_AUDIO_REC.search(line)
        if m:
            audio_ref = m.group(1).strip()
            audio_path = resolve_asset_path(md_file, audio_ref, notebook_root)
            assets["audio"].append(audio_ref)

            blk = new_block(
                blocks, "audio",
                label="AUDIO RECORDING",
                path=audio_ref,
                file_ext=audio_path.suffix.lower().lstrip("."),
                transcript=None,
                transcript_meta=None,
                source_line=idx
            )

            if transcribe:
                if audio_path.exists():
                    transcript, tmeta = transcribe_audio(audio_path, lang="fr")
                    blk["transcript"] = transcript
                    blk["transcript_meta"] = tmeta
                    if transcript is None:
                        errors.append({"file": str(md_file), "line": idx, "kind": "audio_transcription", "detail": tmeta.get("error")})
                else:
                    err = f"Audio file not found: {audio_path}"
                    errors.append({"file": str(md_file), "line": idx, "kind": "audio_missing", "detail": err})
                    blk["transcript_meta"] = {"status": "failed", "error": err}
            continue

        img_match = RE_MD_IMAGE.search(line)
        if img_match:
            img_ref = img_match.group(1).strip()
            img_path = resolve_asset_path(md_file, img_ref, notebook_root)
            assets["images"].append(img_ref)

            img_blk = new_block(
                blocks, "image",
                path=img_ref,
                exists=img_path.exists(),
                source_line=idx
            )
            last_image_id = img_blk["block_id"]

            ocr_m = RE_IMAGE_OCR.search(line)
            if ocr_m:
                ocr_text = (ocr_m.group(1) or "").strip()
                new_block(
                    blocks, "image_ocr",
                    image_block_id=last_image_id,
                    text=ocr_text,
                    marker="IMAGE OCR",
                    source_line=idx
                )
            continue

        ocr_m = RE_IMAGE_OCR.search(line)
        if ocr_m and last_image_id:
            ocr_text = (ocr_m.group(1) or "").strip()
            new_block(
                blocks, "image_ocr",
                image_block_id=last_image_id,
                text=ocr_text,
                marker="IMAGE OCR",
                source_line=idx
            )
            continue

        new_block(blocks, "paragraph", text=line.strip(), source_line=idx)

    page_pack = {
        "metadata": meta,
        "page_id": meta.get("page_id") or md_file.stem,
        "title": meta.get("title") or md_file.stem,
        "notebook": meta.get("notebook"),
        "section": meta.get("section"),
        "source_file": str(md_file),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blocks": blocks,
        "assets": assets,
        "errors": errors
    }
    return page_pack


def main():
    ap = argparse.ArgumentParser(description="Process OneNote markdown exports into LLM-ready JSON page packs.")
    ap.add_argument("notebook", help='Notebook name to process (e.g., "test"). Matches YAML front-matter `notebook:`.')
    ap.add_argument("--input", default="onenote-exporter/output", help="Path to exporter output root.")
    ap.add_argument("--out", default="processed", help="Output folder for processed JSON.")
    ap.add_argument("--transcribe", action="store_true", help="Attempt audio transcription (requires ffmpeg + whisper).")
    ap.add_argument("--copy-assets", action="store_true", help="Copy referenced assets (images/audio) into output folder.")
    args = ap.parse_args()

    input_root = Path(args.input).resolve()
    out_root = Path(args.out).resolve() / args.notebook
    out_pages = out_root / "pages"
    out_assets = out_root / "assets"
    notebook_root = (input_root / args.notebook).resolve()

    out_pages.mkdir(parents=True, exist_ok=True)
    if args.copy_assets:
        (out_assets / "images").mkdir(parents=True, exist_ok=True)
        (out_assets / "audio").mkdir(parents=True, exist_ok=True)

    manifest = {
        "notebook": args.notebook,
        "input_root": str(input_root),
        "output_root": str(out_root),
        "transcribe": bool(args.transcribe),
        "copy_assets": bool(args.copy_assets),
        "processed_pages": [],
        "skipped_files": [],
        "errors": []
    }

    if not input_root.exists():
        print(f"ERROR: input folder not found: {input_root}", file=sys.stderr)
        sys.exit(2)

    md_files = list(input_root.rglob("*.md"))

    for md_file in md_files:
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
            meta, body = parse_frontmatter_and_body(text)

            if (meta.get("notebook") or "").strip() != args.notebook:
                manifest["skipped_files"].append(str(md_file))
                continue

            page_pack = parse_markdown(md_file, meta, body, transcribe=args.transcribe, notebook_root=notebook_root)

            safe_name = (page_pack["page_id"] or md_file.stem).replace("/", "_")
            out_path = out_pages / f"{safe_name}.json"
            out_path.write_text(json.dumps(page_pack, ensure_ascii=False, indent=2), encoding="utf-8")

            manifest["processed_pages"].append({
                "page_id": page_pack["page_id"],
                "title": page_pack["title"],
                "file": str(md_file),
                "out": str(out_path),
                "blocks": len(page_pack["blocks"]),
                "images": len(page_pack["assets"]["images"]),
                "audio": len(page_pack["assets"]["audio"]),
                "errors": len(page_pack["errors"])
            })

            if args.copy_assets:
                for img_ref in page_pack["assets"]["images"]:
                    src = resolve_asset_path(md_file, img_ref)
                    if src.exists():
                        dst = (out_assets / "images" / Path(img_ref).name)
                        shutil.copy2(src, dst)
                for aud_ref in page_pack["assets"]["audio"]:
                    src = resolve_asset_path(md_file, aud_ref)
                    if src.exists():
                        dst = (out_assets / "audio" / Path(aud_ref).name)
                        shutil.copy2(src, dst)

            for e in page_pack["errors"]:
                manifest["errors"].append(e)

        except Exception as e:
            manifest["errors"].append({"file": str(md_file), "kind": "exception", "detail": str(e)})

    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    errlog = out_root / "errors.jsonl"
    with errlog.open("w", encoding="utf-8") as f:
        for e in manifest["errors"]:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    if manifest["errors"]:
        print("\n=== ERROR SUMMARY (first 20) ===")
        for e in manifest["errors"][:20]:
            print(f"- {e.get('kind')} | {e.get('file')} | line={e.get('line')} | {e.get('detail')}")
        print(f"(wrote full log to {errlog})")


    print(json.dumps({
        "notebook": args.notebook,
        "processed_pages": len(manifest["processed_pages"]),
        "skipped_files": len(manifest["skipped_files"]),
        "errors": len(manifest["errors"]),
        "output": str(out_root)
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()