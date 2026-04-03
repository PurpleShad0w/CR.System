import time
from pathlib import Path

import streamlit as st

from pipeline_runner import (
    PipelineConfig,
    OneNoteExportConfig,
    OneNoteProcessConfig,
    LearningPipelineConfig,
    run_pipeline_streaming,
    run_onenote_export_streaming,
    run_onenote_process_streaming,
    run_learning_pipeline_streaming,
    compute_expected_outputs,
)
from utils import load_json, safe_read_text


st.set_page_config(page_title="Build4Use – Interface locale", page_icon="🧩", layout="wide")

project_root_default = Path(__file__).resolve().parent.parent


def _init(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


st.sidebar.title("🧭 Paramètres")
project_root = st.sidebar.text_input(
    "Chemin projet",
    value=str(project_root_default),
    help="Dossier racine contenant process/ et input/ (et run_pipeline.py)",
)
project_root_p = Path(project_root).resolve()

preset_path = Path(__file__).parent / "presets.json"
presets = []
if preset_path.exists():
    try:
        presets = load_json(preset_path).get("presets", [])
    except Exception:
        presets = []
preset_names = [p.get("name", "") for p in presets if p.get("name")]
selected_preset = st.sidebar.selectbox(
    "Profil génération",
    options=(["—"] + preset_names) if preset_names else ["—"],
)


# ---- session defaults ----
_init("gen_notebook", "test")
_init("gen_onenote_section", "Clinique - Goussonville")
_init("gen_case_id", "P060011")
_init("gen_mode", "multistep")
_init("gen_min_quality", 0.0)
_init("gen_bacs_scope", "Non résidentiel")
_init("gen_bacs_part2_slides", False)
_init("gen_bacs_targets", "")
_init(
    "gen_bacs_rules",
    str(project_root_p / "input" / "rules" / "bacs_table6_rules_structured_clean.json"),
)

# IMPORTANT: keep these empty strings; treat empty as "no file" (avoid Path('') == '.')
_init("gen_last_pptx", "")
_init("gen_last_assembled", "")
_init("gen_last_quality", "")

_init("logs_gen", [])

_init("exp_env_path", str(project_root_p / ".env"))
_init("exp_out_dir", str(project_root_p / "input" / "onenote-exporter" / "output"))
_init("exp_token_cache", str(project_root_p / "input" / "onenote-exporter" / "cache" / "token_cache.json"))
_init("exp_formats", "md")
_init("exp_merge", False)
_init("exp_notebook_name", "")
_init("exp_notebook_id", "")
_init("exp_rows", [])
_init("exp_last_click_id", "")
_init("exp_last_click_ts", 0.0)
_init("exp_autorun_doubleclick", True)
_init("exp_autorun_pending", False)
_init("logs_export", [])

_init("proc_input_root", str(project_root_p / "input" / "onenote-exporter" / "output"))
_init("proc_out_root", str(project_root_p / "process" / "onenote"))
_init("proc_transcribe", True)
_init("proc_copy_assets", False)
_init("logs_process", [])

_init("logs_learn", [])


if presets and selected_preset != "—":
    p = next((x for x in presets if x.get("name") == selected_preset), None)
    if p:
        st.session_state["gen_notebook"] = p.get("notebook", st.session_state["gen_notebook"])
        st.session_state["gen_onenote_section"] = p.get("onenote_section", st.session_state["gen_onenote_section"])
        st.session_state["gen_case_id"] = p.get("case_id", st.session_state["gen_case_id"])
        st.session_state["gen_mode"] = p.get("mode", st.session_state["gen_mode"])
        st.session_state["gen_min_quality"] = float(p.get("min_quality", st.session_state["gen_min_quality"]))
        st.session_state["gen_bacs_scope"] = p.get("bacs_building_scope", st.session_state["gen_bacs_scope"])
        st.session_state["gen_bacs_part2_slides"] = bool(p.get("bacs_part2_slides", st.session_state["gen_bacs_part2_slides"]))


st.title("🧩 Build4Use – Interface locale")
st.markdown(
    "Workflow recommandé : **1) Export OneNote** → **2) Traiter l'export** → **3) Générer le rapport** → **4) Learning**"
)

if not (project_root_p / "run_pipeline.py").exists():
    st.warning(
        f"run_pipeline.py introuvable dans: {project_root_p} — corrige le chemin dans la sidebar"
    )


tab1, tab2, tab3 = st.tabs(["📄 Générer un rapport", "📤 Export & Traitement OneNote", "🧠 Learning / Skeletons"])


# ---------------- Tab 1 ----------------
with tab1:
    st.subheader("Génération de rapport")
    colL, colR = st.columns([1, 1])
    with colL:
        st.text_input("Notebook", key="gen_notebook")
        st.text_input("Section OneNote", key="gen_onenote_section")
        st.text_input("Case ID", key="gen_case_id")
    with colR:
        st.selectbox("Mode LLM", options=["multistep", "single"], key="gen_mode")
        st.slider("Seuil qualité (min)", 0.0, 100.0, step=1.0, key="gen_min_quality")

    with st.expander("⚙️ Options BACS", expanded=True):
        st.text_input("Règles Tableau 6", key="gen_bacs_rules")
        st.selectbox("Scope", options=["Non résidentiel", "Résidentiel"], key="gen_bacs_scope")
        st.text_input("Targets (optionnel JSON)", key="gen_bacs_targets")
        st.checkbox("Partie 2 en slides markdown (LLM)", key="gen_bacs_part2_slides")

    status = st.empty()

    with st.expander("📜 Logs (persistants)", expanded=True):
        if st.session_state["logs_gen"]:
            st.code("\n".join(st.session_state["logs_gen"][-500:]), language="text")
            st.download_button(
                "⬇️ Télécharger les logs (txt)",
                data=("\n".join(st.session_state["logs_gen"]) + "\n").encode("utf-8"),
                file_name=f"{st.session_state['gen_case_id']}_render.log.txt",
                mime="text/plain",
            )
        else:
            st.caption("Aucun log pour l'instant.")

    run_btn = st.button("🚀 Générer le rapport", type="primary")

    if run_btn:
        st.session_state["logs_gen"] = []
        status.info("Lancement…")

        cfg = PipelineConfig(
            project_root=project_root_p,
            notebook=st.session_state["gen_notebook"].strip(),
            onenote_section=st.session_state["gen_onenote_section"].strip(),
            case_id=st.session_state["gen_case_id"].strip(),
            mode=st.session_state["gen_mode"],
            min_quality=float(st.session_state["gen_min_quality"]),
            bacs_rules=st.session_state["gen_bacs_rules"].strip(),
            bacs_building_scope=st.session_state["gen_bacs_scope"],
            bacs_targets=st.session_state["gen_bacs_targets"].strip(),
            bacs_part2_slides=bool(st.session_state["gen_bacs_part2_slides"]),
        )

        _, lines = run_pipeline_streaming(cfg)
        for ln in lines:
            st.session_state["logs_gen"].append(ln)
            time.sleep(0.005)

        outs = compute_expected_outputs(project_root_p, cfg.case_id)
        pptx_path = outs["pptx"]

        # Only store paths if they exist (avoid storing empty / '.')
        st.session_state["gen_last_pptx"] = str(pptx_path) if pptx_path.exists() else ""
        st.session_state["gen_last_assembled"] = str(outs["assembled"]) if outs["assembled"].exists() else ""
        st.session_state["gen_last_quality"] = str(outs["quality_report"]) if outs["quality_report"].exists() else ""

        if pptx_path.exists():
            status.success(f"Rapport généré : {pptx_path}")
        else:
            status.warning("PPTX non trouvé — consulte les logs.")

    # Always show last artefacts (safe guards: non-empty + is_file)
    pptx_last_s = (st.session_state.get("gen_last_pptx") or "").strip()
    pptx_last = Path(pptx_last_s) if pptx_last_s else None
    if pptx_last and pptx_last.exists() and pptx_last.is_file():
        st.success(f"Dernier rapport : {pptx_last}")
        try:
            with open(pptx_last, "rb") as f:
                st.download_button(
                    "⬇️ Télécharger le PPTX",
                    data=f.read(),
                    file_name=pptx_last.name,
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
        except PermissionError:
            st.warning("Impossible d'ouvrir le fichier PPTX (PermissionError). Vérifie qu'il n'est pas ouvert dans PowerPoint.")

        col1, col2 = st.columns(2)
        assembled_s = (st.session_state.get("gen_last_assembled") or "").strip()
        quality_s = (st.session_state.get("gen_last_quality") or "").strip()
        assembled_p = Path(assembled_s) if assembled_s else None
        quality_p = Path(quality_s) if quality_s else None
        with col1:
            st.markdown("**assembled_report.json**")
            st.code(safe_read_text(assembled_p) if assembled_p and assembled_p.exists() else "", language="json")
        with col2:
            st.markdown("**quality_report.json**")
            st.code(safe_read_text(quality_p) if quality_p and quality_p.exists() else "", language="json")


# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("Export & Traitement OneNote")
    st.caption(
        "Clique sur une ligne = sélection. Double-clic (rapide) = export immédiat (optionnel). Ensuite, bouton de traitement (process_onenote.py)."
    )

    def parse_notebooks(lines: List[str]) -> List[Dict[str, str]]:
        rows = []
        for ln in lines:
            ln = (ln or "").strip()
            if not ln or "\t" not in ln:
                continue
            name, nid = ln.split("\t", 1)
            name = name.strip(); nid = nid.strip()
            if name and nid:
                rows.append({"Nom": name, "ID": nid})
        seen = set(); out = []
        for r in rows:
            if r["ID"] in seen:
                continue
            seen.add(r["ID"]); out.append(r)
        return out

    colA, colB = st.columns([1, 1])
    with colA:
        st.text_input("Fichier .env", key="exp_env_path")
        st.text_input("Output dir (export)", key="exp_out_dir")
        st.text_input("Token cache", key="exp_token_cache")
        st.text_input("Formats", key="exp_formats")
        st.checkbox("Générer merged.md", key="exp_merge")
        st.checkbox("Double-clic = Export immédiat", key="exp_autorun_doubleclick")
    with colB:
        st.text_input("Notebook (nom, match partiel)", key="exp_notebook_name")
        st.text_input("Notebook ID (exact)", key="exp_notebook_id")

    colX, colY = st.columns([1, 1])
    list_btn = colX.button("📋 Lister les notebooks")
    export_btn = colY.button("📤 Exporter")
    status2 = st.empty()

    with st.expander("📜 Logs export (persistants)", expanded=True):
        if st.session_state["logs_export"]:
            st.code("\n".join(st.session_state["logs_export"][-500:]), language="text")
            st.download_button(
                "⬇️ Télécharger logs export (txt)",
                data=("\n".join(st.session_state["logs_export"]) + "\n").encode("utf-8"),
                file_name="onenote_export.log.txt",
                mime="text/plain",
            )
        else:
            st.caption("Aucun log pour l'instant.")

    def run_export(list_only: bool) -> List[str]:
        st.session_state["logs_export"] = []
        cfg = OneNoteExportConfig(
            project_root=project_root_p,
            env_path=st.session_state["exp_env_path"].strip(),
            list_only=list_only,
            notebook_name=st.session_state["exp_notebook_name"].strip(),
            notebook_id=st.session_state["exp_notebook_id"].strip(),
            merge=bool(st.session_state["exp_merge"]),
            formats=st.session_state["exp_formats"].strip(),
            output_dir=st.session_state["exp_out_dir"].strip(),
            token_cache=st.session_state["exp_token_cache"].strip(),
        )
        status2.info("Lancement export…")
        _, lines = run_onenote_export_streaming(cfg)
        collected = []
        for ln in lines:
            collected.append(ln)
            st.session_state["logs_export"].append(ln)
            time.sleep(0.005)
        return collected

    if list_btn:
        lines = run_export(True)
        st.session_state["exp_rows"] = parse_notebooks(lines)
        if st.session_state["exp_rows"]:
            status2.success(f"Notebooks détectés : {len(st.session_state['exp_rows'])}")
        else:
            status2.warning("Aucun notebook détecté. Vérifie .env / permissions.")

    if export_btn:
        run_export(False)
        status2.success("Export terminé (voir logs).")

    st.markdown("---")
    st.subheader("⚙️ Traiter l’export (process_onenote.py)")
    colP1, colP2 = st.columns([1, 1])
    with colP1:
        proc_notebook_default = st.session_state.get("exp_notebook_name") or st.session_state.get("gen_notebook")
        proc_notebook = st.text_input("Notebook à traiter", value=proc_notebook_default)
        st.text_input("Input root", key="proc_input_root")
        st.text_input("Output root", key="proc_out_root")
    with colP2:
        st.checkbox("Transcrire audio (--transcribe)", key="proc_transcribe")
        st.checkbox("Copier assets (--copy-assets)", key="proc_copy_assets")

    process_btn = st.button("➡️ Lancer le traitement", type="secondary")    
    statusP = st.empty()

    with st.expander("📜 Logs traitement (persistants)", expanded=True):
        if st.session_state["logs_process"]:
            st.code("\n".join(st.session_state["logs_process"][-600:]), language="text")
            st.download_button(
                "⬇️ Télécharger logs traitement (txt)",
                data=("\n".join(st.session_state["logs_process"]) + "\n").encode("utf-8"),
                file_name="onenote_process.log.txt",
                mime="text/plain",
            )
        else:
            st.caption("Aucun log pour l'instant.")

    if process_btn:
        st.session_state["logs_process"] = []
        cfgp = OneNoteProcessConfig(
            project_root=project_root_p,
            notebook=proc_notebook.strip(),
            input_root=st.session_state["proc_input_root"].strip(),
            out_root=st.session_state["proc_out_root"].strip(),
            transcribe=bool(st.session_state["proc_transcribe"]),
            copy_assets=bool(st.session_state["proc_copy_assets"]),
        )
        statusP.info("Traitement en cours…")
        _, lines = run_onenote_process_streaming(cfgp)
        for ln in lines:
            st.session_state["logs_process"].append(ln)
            time.sleep(0.005)
        statusP.success("Traitement terminé (voir logs).")


# ---------------- Tab 3 ----------------
with tab3:
    st.subheader("Learning pipeline (corpus + skeletons)")
    st.caption("Exécute run_learning_pipeline.py : process_reports.py puis build_skeletons.py")

    run_learn = st.button("🧠 Lancer learning pipeline")
    status3 = st.empty()

    with st.expander("📜 Logs learning (persistants)", expanded=True):
        if st.session_state["logs_learn"]:
            st.code("\n".join(st.session_state["logs_learn"][-600:]), language="text")
            st.download_button(
                "⬇️ Télécharger logs learning (txt)",
                data=("\n".join(st.session_state["logs_learn"]) + "\n").encode("utf-8"),
                file_name="learning.log.txt",
                mime="text/plain",
            )
        else:
            st.caption("Aucun log pour l'instant.")

    if run_learn:
        st.session_state["logs_learn"] = []
        cfg = LearningPipelineConfig(project_root=project_root_p)
        status3.info("Lancement…")
        _, lines = run_learning_pipeline_streaming(cfg)
        for ln in lines:
            st.session_state["logs_learn"].append(ln)
            time.sleep(0.005)
        status3.success("Learning terminé (voir logs).")
