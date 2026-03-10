import json
import time
from pathlib import Path

import streamlit as st

from pipeline_runner import PipelineConfig, run_streaming, compute_expected_outputs
from utils import load_json, safe_read_text


st.set_page_config(
    page_title="Build4Use – Générateur de rapports (local)",
    page_icon="🧩",
    layout="wide",
)

# --- Styling ---
st.markdown(
    """
<style>
:root {
  --b4u-blue: #0A3D62;
  --b4u-accent: #1B9CFC;
  --b4u-bg: #0b1220;
}
.block-container { padding-top: 1.25rem; }
div[data-testid="stSidebar"] { background: #0b1220; }
div[data-testid="stSidebar"] * { color: #E8EEF6; }
h1, h2, h3 { color: #0A3D62; }
.kpi { background: #f5f8ff; border: 1px solid #e6ecff; padding: 12px 14px; border-radius: 10px; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
.small { font-size: 0.9rem; color: #5b6b7f; }
hr { border: none; border-top: 1px solid #e8eef6; margin: 1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

project_root_default = Path('.').resolve()

# --- Sidebar: Presets & Config ---
st.sidebar.title("🧭 Paramètres")

preset_path = Path(__file__).parent / 'presets.json'
presets = None
if preset_path.exists():
    presets = load_json(preset_path).get('presets', [])

preset_names = [p['name'] for p in presets] if presets else []
selected_preset = st.sidebar.selectbox("Profil", options=(['—'] + preset_names) if preset_names else ['—'])

# Defaults
notebook = "test"
onenote_section = ""
case_id = "P050011"
mode = "multistep"
min_quality = 0.0
bacs_building_scope = "Non résidentiel"
bacs_part2_slides = False

# Apply preset
if presets and selected_preset != '—':
    p = next(x for x in presets if x['name'] == selected_preset)
    notebook = p.get('notebook', notebook)
    onenote_section = p.get('onenote_section', onenote_section)
    case_id = p.get('case_id', case_id)
    mode = p.get('mode', mode)
    min_quality = float(p.get('min_quality', min_quality))
    bacs_building_scope = p.get('bacs_building_scope', bacs_building_scope)
    bacs_part2_slides = bool(p.get('bacs_part2_slides', bacs_part2_slides))

st.sidebar.subheader("Données OneNote")
notebook = st.sidebar.text_input("Notebook", value=notebook)
onenote_section = st.sidebar.text_input("Section (optionnel)", value=onenote_section, help="Laisser vide pour le mode 'case-id'.")
case_id = st.sidebar.text_input("Case ID", value=case_id, help="Si 'Section' est renseignée, le pipeline peut choisir un slug automatiquement; ce champ reste utile en mode case-id.")

st.sidebar.subheader("Génération")
mode = st.sidebar.selectbox("Mode LLM", options=["multistep", "single"], index=0 if mode == 'multistep' else 1)
min_quality = st.sidebar.slider("Seuil qualité (min)", min_value=0.0, max_value=100.0, value=float(min_quality), step=1.0)

st.sidebar.subheader("BACS / ISO 52120-1")
# default rules path in project
rules_default = str(project_root_default / 'input' / 'rules' / 'bacs_table6_rules_structured_clean.json')
bacs_rules = st.sidebar.text_input("Fichier règles (Tableau 6)", value=rules_default)
bacs_building_scope = st.sidebar.selectbox("Scope", options=["Non résidentiel", "Résidentiel"], index=0 if bacs_building_scope == 'Non résidentiel' else 1)
bacs_targets = st.sidebar.text_input("Targets (optionnel JSON)", value="", help="Chemin vers un JSON mapping groupe→classe cible.")
bacs_part2_slides = st.sidebar.checkbox("Partie 2 en 'slides markdown' (LLM)", value=bacs_part2_slides)

st.sidebar.subheader("Projet")
project_root = st.sidebar.text_input("Chemin projet", value=str(project_root_default))

# --- Main ---
st.title("🧩 Build4Use – Générateur de rapports (local)")
st.markdown("""<div class='small'>Objectif : permettre à des collègues non-développeurs de lancer la pipeline via une interface web locale.</div>""", unsafe_allow_html=True)

colA, colB, colC = st.columns([1.2, 1.2, 1.6])

with colA:
    st.markdown("<div class='kpi'><b>Entrée</b><br/>OneNote → pages exportées → pipeline</div>", unsafe_allow_html=True)
with colB:
    st.markdown("<div class='kpi'><b>Sortie</b><br/>PPTX + JSON (assembled, quality)</div>", unsafe_allow_html=True)
with colC:
    st.markdown("<div class='kpi'><b>Astuce</b><br/>Renseigne la section OneNote si tu veux restreindre l'analyse à un site précis.</div>", unsafe_allow_html=True)

st.markdown("---")

run_btn = st.button("🚀 Générer le rapport", type="primary")

log_box = st.empty()
status = st.empty()

if 'logs' not in st.session_state:
    st.session_state.logs = []

if run_btn:
    st.session_state.logs = []

    cfg = PipelineConfig(
        project_root=Path(project_root).resolve(),
        notebook=notebook.strip(),
        onenote_section=onenote_section.strip(),
        case_id=case_id.strip(),
        mode=mode,
        min_quality=float(min_quality),
        bacs_rules=bacs_rules.strip(),
        bacs_building_scope=bacs_building_scope,
        bacs_targets=bacs_targets.strip(),
        bacs_part2_slides=bool(bacs_part2_slides),
    )

    status.info("Lancement de la pipeline…")

    pid, lines = run_streaming(cfg)

    for ln in lines:
        st.session_state.logs.append(ln)
        # keep last ~400 lines for UI
        view = "\n".join(st.session_state.logs[-400:])
        log_box.code(view, language="text")
        if ln.strip().startswith('❌'):
            status.error(ln)
        elif ln.strip().startswith('✅'):
            status.success(ln)
        elif ln.strip().startswith('▶'):
            status.info(ln)
        else:
            # keep last status line visible
            pass
        time.sleep(0.01)

    # Post-run outputs
    outs = compute_expected_outputs(cfg.project_root, cfg.case_id)
    pptx_path = outs['pptx']

    if pptx_path.exists():
        status.success(f"Rapport généré : {pptx_path}")
        st.subheader("📄 Résultats")
        st.write(f"PPTX : `{pptx_path}`")

        with open(pptx_path, 'rb') as f:
            st.download_button(
                label="⬇️ Télécharger le PPTX",
                data=f.read(),
                file_name=pptx_path.name,
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**assembled_report.json**")
            st.code(safe_read_text(outs['assembled'], limit=80_000), language='json')
        with col2:
            st.markdown("**quality_report.json**")
            st.code(safe_read_text(outs['quality_report'], limit=80_000), language='json')

    else:
        status.warning("Aucun PPTX trouvé à l'emplacement attendu. Consulte les logs ci-dessus.")

st.markdown("---")

with st.expander("ℹ️ Aide & utilisation", expanded=False):
    st.markdown(
        """
**Pré-requis**
- Python installé sur le poste.
- Le projet (ce dépôt) accessible localement.

**Lancer l'interface**
- Windows: double-clique sur `start.bat`
- macOS/Linux: `bash start.sh`

**Organisation**
- Cette UI appelle `run_pipeline.py` et affiche les logs en direct.
- Les paramètres visibles dans la sidebar sont volontairement limités.

**Modularité**
- Les profils se modifient dans `presets.json`.
"""
    )
