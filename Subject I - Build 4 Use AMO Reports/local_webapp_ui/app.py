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

st.markdown(
    """
<style>
.block-container { padding-top: 1.25rem; }
div[data-testid="stSidebar"] { background: #0b1220; }
div[data-testid="stSidebar"] * { color: #E8EEF6; }
h1, h2, h3 { color: #0A3D62; }
.kpi { background: #f5f8ff; border: 1px solid #e6ecff; padding: 12px 14px; border-radius: 10px; }
.small { font-size: 0.9rem; color: #5b6b7f; }
hr { border: none; border-top: 1px solid #e8eef6; margin: 1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# IMPORTANT FIX:
# The UI is located in ./local_webapp_ui but run_pipeline.py lives at repo root.
# Default project root must therefore be the parent folder of this UI directory.
project_root_default = Path(__file__).resolve().parent.parent

st.sidebar.title("🧭 Paramètres")

preset_path = Path(__file__).parent / 'presets.json'
presets = load_json(preset_path).get('presets', []) if preset_path.exists() else []
preset_names = [p['name'] for p in presets]
selected_preset = st.sidebar.selectbox("Profil", options=(['—'] + preset_names) if preset_names else ['—'])

# Defaults
notebook = "test"
onenote_section = ""
case_id = "P050011"
mode = "multistep"
min_quality = 0.0
bacs_building_scope = "Non résidentiel"
bacs_part2_slides = False

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
case_id = st.sidebar.text_input("Case ID", value=case_id)

st.sidebar.subheader("Génération")
mode = st.sidebar.selectbox("Mode LLM", options=["multistep", "single"], index=0 if mode == 'multistep' else 1)
min_quality = st.sidebar.slider("Seuil qualité (min)", min_value=0.0, max_value=100.0, value=float(min_quality), step=1.0)

st.sidebar.subheader("BACS / ISO 52120-1")
rules_default = str(project_root_default / 'input' / 'rules' / 'bacs_table6_rules_structured_clean.json')
bacs_rules = st.sidebar.text_input("Fichier règles (Tableau 6)", value=rules_default)
bacs_building_scope = st.sidebar.selectbox("Scope", options=["Non résidentiel", "Résidentiel"], index=0 if bacs_building_scope == 'Non résidentiel' else 1)
bacs_targets = st.sidebar.text_input("Targets (optionnel JSON)", value="")
bacs_part2_slides = st.sidebar.checkbox("Partie 2 en 'slides markdown' (LLM)", value=bacs_part2_slides)

st.sidebar.subheader("Projet")
project_root = st.sidebar.text_input("Chemin projet", value=str(project_root_default), help="Doit pointer sur le dossier qui contient run_pipeline.py")

st.title("🧩 Build4Use – Générateur de rapports (local)")
st.markdown("""<div class='small'>Interface web locale pour lancer la pipeline via un bouton (sans Python pour l'utilisateur).</div>""", unsafe_allow_html=True)

colA, colB, colC = st.columns([1.2, 1.2, 1.6])
with colA:
    st.markdown("<div class='kpi'><b>Entrée</b><br/>OneNote → pipeline</div>", unsafe_allow_html=True)
with colB:
    st.markdown("<div class='kpi'><b>Sortie</b><br/>PPTX + JSON</div>", unsafe_allow_html=True)
with colC:
    st.markdown("<div class='kpi'><b>Astuce</b><br/>Si une section est renseignée, le rapport se limite à ce site.</div>", unsafe_allow_html=True)

st.markdown('---')

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

    # quick sanity check for common misconfig
    rp = cfg.project_root / 'run_pipeline.py'
    if not rp.exists():
        status.error(f"run_pipeline.py introuvable dans: {cfg.project_root} — corrige 'Chemin projet' dans la sidebar")
    else:
        status.info("Lancement de la pipeline…")
        pid, lines = run_streaming(cfg)

        for ln in lines:
            st.session_state.logs.append(ln)
            view = "\n".join(st.session_state.logs[-400:])
            log_box.code(view, language='text')
            if ln.strip().startswith('❌'):
                status.error(ln)
            elif ln.strip().startswith('✅'):
                status.success(ln)
            elif ln.strip().startswith('▶'):
                status.info(ln)
            time.sleep(0.01)

        outs = compute_expected_outputs(cfg.project_root, cfg.case_id)
        pptx_path = outs['pptx']

        if pptx_path.exists():
            status.success(f"Rapport généré : {pptx_path}")
            st.subheader("📄 Résultats")
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
            status.warning("Aucun PPTX trouvé à l'emplacement attendu. Consulte les logs.")

with st.expander("ℹ️ Aide & utilisation"):
    st.markdown(
        """
**But** : permettre aux collègues de lancer la génération sans Python.

**Dépannage rapide** :
- Si le message dit que `run_pipeline.py` est introuvable, vérifie le champ **Chemin projet** (il doit pointer sur la racine du repo).
"""
    )
