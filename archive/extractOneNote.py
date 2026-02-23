import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone


SUPPORTED_SUFFIXES = {
	".one",
	".onepart",
	".one.onepart",
	".onetoc2"
}

LOG_STDOUT_NAME = "stdout.txt"
LOG_STDERR_NAME = "stderr.txt"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
TEXT_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")

NODE_HEADER_RE = re.compile(r"^\s*(jcid\w+)\(<ExtendedGUID>\s*\(([0-9a-fA-F-]+),\s*(\d+)\)\)\s*:\s*$")
PROP_RE = re.compile(r"^\s+([A-Za-z0-9_]+)\s*:\s*(.*)$")
EXGUID_IN_LIST_RE = re.compile(r"<ExtendedGUID>\s*\(([0-9a-fA-F-]+),\s*(\d+)\)")


def sha256_file(path: Path) -> str:
	h = hashlib.sha256()
	with path.open("rb") as f:
		for chunk in iter(lambda: f.read(1024 * 1024), b""):
			h.update(chunk)
	return h.hexdigest()


def is_supported_onenote_file(p: Path) -> bool:
	name = p.name.lower()
	return any(name.endswith(suf) for suf in SUPPORTED_SUFFIXES)


def html_escape(s: str) -> str:
	return (s.replace("&", "&amp;")
		.replace("<", "&lt;")
		.replace(">", "&gt;"))


def safe_filename(name: str) -> str:
	name = (name or "").replace("\x00", "").strip()
	name = re.sub(r"[<>:\"/\\|?*\n\r\t]+", "_", name)
	name = re.sub(r"\s+", " ", name).strip()
	return name[:180] if name else "unnamed"


def exguid_key(guid: str, n: str | int) -> str:
	return f"{guid.lower()},{int(n)}"


def guid_of_key(key: str) -> str:
	return key.split(",", 1)[0].lower()


def run_pyonenote(input_file: Path, out_dir: Path, assets_dir: Path, extension: str | None = None):
	env = os.environ.copy()
	env["PYTHONUTF8"] = "1"
	env["PYTHONIOENCODING"] = "utf-8"

	out_dir.mkdir(parents=True, exist_ok=True)
	assets_dir.mkdir(parents=True, exist_ok=True)

	log_out = out_dir / LOG_STDOUT_NAME
	log_err = out_dir / LOG_STDERR_NAME

	cmd = [sys.executable, "-m", "pyOneNote.Main", "-f", str(input_file), "-o", str(assets_dir)]
	if extension:
		cmd += ["-e", extension]

	with log_out.open("w", encoding="utf-8", errors="replace") as fo, \
		log_err.open("w", encoding="utf-8", errors="replace") as fe:
		proc = subprocess.run(cmd, stdout=fo, stderr=fe, env=env)

	return proc.returncode, log_out, log_err


def collect_assets(assets_dir: Path) -> list:
	assets = []
	if not assets_dir.exists():
		return assets
	for f in sorted(assets_dir.rglob("*")):
		if not f.is_file():
			continue
		rel = str(f.relative_to(assets_dir))
		assets.append({
			"relative_path": f"assets/{rel}",
			"filename": f.name,
			"bytes": f.stat().st_size,
			"sha256": sha256_file(f),
		})
	return assets


def parse_dump_to_nodes(stdout_text: str) -> dict:
	nodes = {}
	cur = None
	order = 0

	for line in stdout_text.splitlines():
		mh = NODE_HEADER_RE.match(line)
		if mh:
			ntype, guid, idx = mh.groups()
			cur = exguid_key(guid, idx)
			nodes[cur] = {"type": ntype, "props": {}, "order": order}
			order += 1
			continue

		if not cur:
			continue

		mp = PROP_RE.match(line)
		if not mp:
			continue

		pname, raw = mp.groups()
		raw = raw.replace("\x00", "").strip()

		if raw.startswith("[") and "<ExtendedGUID>" in raw:
			children = [exguid_key(g, i) for g, i in EXGUID_IN_LIST_RE.findall(raw)]
			nodes[cur]["props"][pname] = children
		else:
			nodes[cur]["props"][pname] = raw

	return nodes


def get_children(nodes: dict, key: str, *prop_names: str) -> list:
	props = nodes.get(key, {}).get("props", {})
	for pn in prop_names:
		v = props.get(pn)
		if isinstance(v, list) and v:
			return v
	return []


def find_page_nodes(nodes: dict) -> list:
	pages = [k for k, v in nodes.items() if v.get("type") == "jcidPageNode"]
	if pages:
		return pages
	roots = []
	for k, v in nodes.items():
		if v.get("type") == "jcidPageSeriesNode":
			roots.extend(get_children(nodes, k, "ChildGraphSpaceElementNodes"))
	return roots


def get_float(props: dict, name: str, default=float("inf")) -> float:
	try:
		return float(props.get(name))
	except Exception:
		return default


def pos_key(nodes: dict, key: str):
	props = nodes.get(key, {}).get("props", {})
	y = get_float(props, "OffsetFromParentVert")
	x = get_float(props, "OffsetFromParentHoriz")
	o = nodes.get(key, {}).get("order", 10**12)
	return (y, x, o)


def sort_by_pos(nodes: dict, keys: list) -> list:
	return sorted(keys, key=lambda k: pos_key(nodes, k))


def dedupe_preserve_order(items: list) -> list:
	seen = set()
	out = []
	for x in items:
		if x in seen:
			continue
		seen.add(x)
		out.append(x)
	return out


def looks_mojibake(s: str) -> bool:
	if not s:
		return False
	cjk = 0
	latin = 0
	for ch in s:
		o = ord(ch)
		if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
			cjk += 1
		elif (0x20 <= o <= 0x7E) or (0x00C0 <= o <= 0x024F):
			latin += 1
	return cjk >= 4 and cjk > latin * 2

def fix_mojibake(s: str) -> str:
	s = (s or "").replace("\x00", "").strip()
	if not s or not looks_mojibake(s):
		return s

	candidates = []

	try:
		candidates.append(s.encode("utf-16le").decode("utf-8", errors="strict"))
	except Exception:
		pass
	try:
		candidates.append(s.encode("utf-16be").decode("utf-8", errors="strict"))
	except Exception:
		pass
	try:
		candidates.append(s.encode("utf-16le").decode("latin-1", errors="strict"))
	except Exception:
		pass
	try:
		candidates.append(s.encode("utf-16be").decode("latin-1", errors="strict"))
	except Exception:
		pass

	def score(t: str) -> int:
		if not t:
			return -10**9
		ok = 0
		bad = 0
		for ch in t:
			o = ord(ch)
			if (0x20 <= o <= 0x7E) or (0x00C0 <= o <= 0x024F):
				ok += 1
			elif o < 0x20 and ch not in "\n\r\t":
				bad += 2
			elif 0x4E00 <= o <= 0x9FFF:
				bad += 3
		return ok - bad

	best = s
	best_score = score(s)
	for c in candidates:
		sc = score(c)
		if sc > best_score:
			best = c
			best_score = sc

	return best


def decode_text_extended_ascii(val: str) -> str:
	val = (val or "").replace("\x00", "").strip()
	if not val:
		return ""
	if TEXT_HEX_RE.match(val) and len(val) % 2 == 0:
		try:
			b = bytes.fromhex(val)
		except Exception:
			return val
		for enc in ("utf-8", "latin-1", "utf-16le"):
			try:
				txt = b.decode(enc, errors="strict")
				txt = fix_mojibake(txt)
				if txt.strip():
					return txt
			except Exception:
				pass
		txt = b.decode("utf-8", errors="replace")
		return fix_mojibake(txt)
	return fix_mojibake(val)


def get_richtext_text(props: dict) -> str:
	u = (props.get("RichEditTextUnicode", "") or "").replace("\x00", "").strip()
	u = fix_mojibake(u)
	if u:
		return u
	tea = props.get("TextExtendedAscii", "")
	t = decode_text_extended_ascii(tea)
	if t:
		return t
	return ""


def resolve_content_key(nodes: dict, start_key: str, max_hops: int = 10) -> str | None:
	key = start_key
	seen = set()
	for _ in range(max_hops):
		if not key or key in seen:
			return None
		seen.add(key)
		ntype = nodes.get(key, {}).get("type", "")
		props = nodes.get(key, {}).get("props", {})
		if ntype in ("jcidRichTextOENode", "jcidImageNode", "jcidEmbeddedFileNode", "jcidTableNode"):
			return key
		next_keys = []
		for pn in ("ContentChildNodesOfOutlineElement", "ContentChildNodes", "ContentChildNodesOfPageManifest", "ElementChildNodes", "ElementChildNodesOfPageManifest"):
			v = props.get(pn)
			if isinstance(v, list) and v:
				next_keys = v
				break
		if not next_keys:
			return None
		key = next_keys[0]
	return None


def build_media_name_lists(stdout_text: str):
	image_names = []
	embed_names = []

	cur_type = None
	cur_image_name = None
	cur_embed_name = None

	for line in stdout_text.splitlines():
		mh = NODE_HEADER_RE.match(line)
		if mh:
			if cur_type == "jcidImageNode" and cur_image_name:
				image_names.append(cur_image_name)
			if cur_type == "jcidEmbeddedFileNode" and cur_embed_name:
				embed_names.append(cur_embed_name)
			cur_type = mh.group(1)
			cur_image_name = None
			cur_embed_name = None
			continue

		if not cur_type:
			continue

		mp = PROP_RE.match(line)
		if not mp:
			continue

		pname, raw = mp.groups()
		raw = raw.replace("\x00", "").strip()

		if cur_type == "jcidImageNode" and pname == "ImageFilename":
			cur_image_name = raw
		if cur_type == "jcidEmbeddedFileNode" and pname == "EmbeddedFileName":
			cur_embed_name = raw

	if cur_type == "jcidImageNode" and cur_image_name:
		image_names.append(cur_image_name)
	if cur_type == "jcidEmbeddedFileNode" and cur_embed_name:
		embed_names.append(cur_embed_name)

	return image_names, embed_names


def build_named_assets(assets_dir: Path, stdout_text: str) -> dict:
	named_dir = assets_dir / "named"
	if named_dir.exists():
		shutil.rmtree(named_dir)
	named_dir.mkdir(parents=True, exist_ok=True)

	image_names, embed_names = build_media_name_lists(stdout_text)

	extracted = [p for p in sorted(assets_dir.iterdir()) if p.is_file()]
	extracted = [p for p in extracted if p.name.lower() not in {LOG_STDOUT_NAME, LOG_STDERR_NAME}]
	extracted_images = [p for p in extracted if p.suffix.lower() in IMAGE_EXTS]
	extracted_other = [p for p in extracted if p.suffix.lower() not in IMAGE_EXTS]

	mapping_images = {}
	mapping_files = {}

	for i, nm in enumerate(image_names):
		if i >= len(extracted_images):
			break
		target_name = safe_filename(nm)
		src = extracted_images[i]
		if Path(target_name).suffix == "":
			target_name = f"{target_name}{src.suffix.lower()}"
		dst = named_dir / target_name
		if dst.exists():
			base = dst.stem
			ext = dst.suffix
			k = 2
			while (named_dir / f"{base}_{k}{ext}").exists():
				k += 1
			dst = named_dir / f"{base}_{k}{ext}"
		shutil.copy2(src, dst)
		mapping_images[nm] = f"assets/named/{dst.name}"

	for i, nm in enumerate(embed_names):
		if i >= len(extracted_other):
			break
		target_name = safe_filename(nm)
		src = extracted_other[i]
		if Path(target_name).suffix == "" and src.suffix:
			target_name = f"{target_name}{src.suffix.lower()}"
		dst = named_dir / target_name
		if dst.exists():
			base = dst.stem
			ext = dst.suffix
			k = 2
			while (named_dir / f"{base}_{k}{ext}").exists():
				k += 1
			dst = named_dir / f"{base}_{k}{ext}"
		shutil.copy2(src, dst)
		mapping_files[nm] = f"assets/named/{dst.name}"

	return {
		"images": mapping_images,
		"files": mapping_files,
	}


def emit_node_as_html(nodes: dict, key: str, media_map: dict) -> list:
	out = []
	ctype = nodes.get(key, {}).get("type", "")
	props = nodes.get(key, {}).get("props", {})

	if ctype == "jcidRichTextOENode":
		txt = get_richtext_text(props)
		if txt:
			out.append(f"<p>{html_escape(txt)}</p>")
		return out

	if ctype == "jcidImageNode":
		img_name = (props.get("ImageFilename", "") or "").replace("\x00", "").strip()
		alt = (props.get("ImageAltText", "") or "").replace("\x00", "").strip()
		ref = media_map.get("images", {}).get(img_name)
		if img_name:
			if ref:
				if alt:
					out.append(f"<p>⟦IMAGE:{html_escape(img_name)}|{html_escape(ref)}|ALT:{html_escape(alt)}⟧</p>")
				else:
					out.append(f"<p>⟦IMAGE:{html_escape(img_name)}|{html_escape(ref)}⟧</p>")
			else:
				out.append(f"<p>⟦IMAGE:{html_escape(img_name)}⟧</p>")
		return out

	if ctype == "jcidEmbeddedFileNode":
		fname = (props.get("EmbeddedFileName", "") or props.get("RichEditTextUnicode", "") or "").replace("\x00", "").strip()
		ref = media_map.get("files", {}).get(fname)
		if fname:
			if ref:
				out.append(f"<p>⟦FILE:{html_escape(fname)}|{html_escape(ref)}⟧</p>")
			else:
				out.append(f"<p>⟦FILE:{html_escape(fname)}⟧</p>")
		return out

	if ctype == "jcidTableNode":
		out.append(f"<p>⟦TABLE:{html_escape(key)}⟧</p>")
		return out

	return out


def render_outline_element(nodes: dict, elem_key: str, media_map: dict) -> list:
	out = []
	content_keys = get_children(nodes, elem_key, "ContentChildNodesOfOutlineElement", "ContentChildNodes", "ContentChildNodesOfPageManifest")
	if not content_keys:
		return out
	raw_ck = content_keys[0]
	ck = resolve_content_key(nodes, raw_ck)
	if not ck:
		return out
	return emit_node_as_html(nodes, ck, media_map)


def render_outline(nodes: dict, outline_key: str, media_map: dict) -> list:
	out = []
	elems = get_children(nodes, outline_key, "ElementChildNodesOfOutline", "ElementChildNodes")
	if not elems:
		elems = get_children(nodes, outline_key, "ElementChildNodesOfVersionHistory")
	if not elems:
		return out

	elems = dedupe_preserve_order(elems)
	elems = [k for k in elems if nodes.get(k, {}).get("type") == "jcidOutlineElementNode"]

	seen = set()
	for ek in elems:
		for line in render_outline_element(nodes, ek, media_map):
			if line in seen:
				continue
			seen.add(line)
			out.append(line)

	return out


def render_page_linked(nodes: dict, page_key: str, media_map: dict) -> list:
	out = []
	page_children = get_children(nodes, page_key, "ElementChildNodesOfPage", "ElementChildNodes")
	if not page_children:
		page_children = get_children(nodes, page_key, "StructureElementChildNodes")
	if not page_children:
		return out

	page_children = dedupe_preserve_order(page_children)

	for bk in page_children:
		btype = nodes.get(bk, {}).get("type", "")

		if btype == "jcidOutlineNode":
			out.extend(render_outline(nodes, bk, media_map))
			continue

		if btype == "jcidTitleNode":
			title_children = get_children(nodes, bk, "ElementChildNodesOfTitle", "ElementChildNodes")
			if not title_children:
				title_children = get_children(nodes, bk, "ElementChildNodesOfVersionHistory")
			title_children = dedupe_preserve_order(title_children)
			for tk in title_children:
				if nodes.get(tk, {}).get("type") == "jcidOutlineNode":
					out.extend(render_outline(nodes, tk, media_map))
			continue

		if btype in ("jcidImageNode", "jcidEmbeddedFileNode", "jcidRichTextOENode"):
			out.extend(emit_node_as_html(nodes, bk, media_map))
			continue

	return out


def render_page_fallback(nodes: dict, page_key: str, media_map: dict) -> list:
	pg = guid_of_key(page_key)
	candidates = []
	for k, v in nodes.items():
		if guid_of_key(k) != pg:
			continue
		t = v.get("type", "")
		if t in ("jcidRichTextOENode", "jcidImageNode", "jcidEmbeddedFileNode", "jcidTableNode"):
			candidates.append(k)

	if not candidates:
		return []

	candidates.sort(key=lambda k: nodes.get(k, {}).get("order", 10**12))

	out = []
	seen = set()
	for k in candidates:
		for line in emit_node_as_html(nodes, k, media_map):
			if not line:
				continue
			if line in seen:
				continue
			seen.add(line)
			out.append(line)
	return out


def reconstruct_content_html_ordered(stdout_text: str, media_map: dict) -> str:
	nodes = parse_dump_to_nodes(stdout_text)
	page_keys = find_page_nodes(nodes)

	html_lines = []
	for pk in page_keys:
		linked = render_page_linked(nodes, pk, media_map)
		has_text = any(line.startswith("<p>") and "⟦IMAGE:" not in line and "⟦FILE:" not in line and "⟦TABLE:" not in line for line in linked)
		if not has_text:
			fallback = render_page_fallback(nodes, pk, media_map)
			if fallback:
				linked = fallback
		html_lines.extend(linked)

	return "\n".join(html_lines).strip()


def build_record(input_file: Path, out_dir: Path, assets: list, rc: int, stdout_log: Path, stderr_log: Path, content_html: str, media_map: dict) -> dict:
	return {
		"source": {
			"path": str(input_file),
			"name": input_file.name,
			"bytes": input_file.stat().st_size if input_file.exists() else None,
			"sha256": sha256_file(input_file) if input_file.exists() else None,
		},
		"run": {
			"utc": datetime.now(timezone.utc).isoformat(),
			"output_dir": str(out_dir),
		},
		"pyonenote": {
			"returncode": rc,
			"stdout_log": str(stdout_log),
			"stderr_log": str(stderr_log),
		},
		"content_html": content_html,
		"assets": assets,
		"media_map": media_map,
	}


def main():
	ap = argparse.ArgumentParser(description="Extract OneNote + reconstruct HTML with stable media and fallback text recovery.")
	ap.add_argument("input_folder", help="Folder containing OneNote files.")
	ap.add_argument("output_folder", help="Folder to write extracted content into.")
	ap.add_argument("--extension", default=None, help="Optional extension to append to extracted files (pyonenote -e).")
	ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
	ap.add_argument("--skip-existing", action="store_true", help="Skip files that already have a record.json in their output directory.")
	args = ap.parse_args()

	in_root = Path(args.input_folder).expanduser().resolve()
	out_root = Path(args.output_folder).expanduser().resolve()
	out_root.mkdir(parents=True, exist_ok=True)

	if not in_root.exists() or not in_root.is_dir():
		print(f"Input folder not found or not a directory: {in_root}", file=sys.stderr)
		return 2

	iterator = in_root.rglob("*") if args.recursive else in_root.glob("*")
	targets = [p for p in iterator if p.is_file() and is_supported_onenote_file(p)]

	if not targets:
		print("No supported OneNote files found.")
		return 0

	print(f"Found {len(targets)} OneNote file(s).")

	index = []
	failures = 0

	for f in targets:
		safe_name = f.name.replace(os.sep, "_")
		per_file_out = out_root / safe_name
		per_file_out.mkdir(parents=True, exist_ok=True)

		assets_dir = per_file_out / "assets"
		record_path = per_file_out / "record.json"
		reconstructed_path = per_file_out / "reconstructed.html"

		if args.skip_existing and record_path.exists():
			print(f"[SKIP]\t{f}")
			continue

		rc, stdout_log, stderr_log = run_pyonenote(f, per_file_out, assets_dir, args.extension)

		if rc != 0:
			failures += 1
			print(f"[FAIL]\t{f}\t(rc={rc})")
		else:
			print(f"[OK]\t{f}")

		try:
			stdout_text = stdout_log.read_text(encoding="utf-8", errors="replace")
		except Exception:
			stdout_text = ""

		media_map = build_named_assets(assets_dir, stdout_text)
		assets = collect_assets(assets_dir)

		content_html = reconstruct_content_html_ordered(stdout_text, media_map)
		reconstructed_path.write_text(content_html, encoding="utf-8", errors="replace")

		record = build_record(f, per_file_out, assets, rc, stdout_log, stderr_log, content_html, media_map)
		record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

		index.append({
			"source_file": str(f),
			"output_dir": str(per_file_out),
			"record": str(record_path),
			"reconstructed": str(reconstructed_path),
			"assets_dir": str(assets_dir),
			"assets_count": len(assets),
			"returncode": rc,
		})

	index_path = out_root / "index.json"
	index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

	print(f"Index written to:\t{index_path}")
	print(f"Failures:\t{failures}")
	return 1 if failures else 0


if __name__ == "__main__":
	raise SystemExit(main())