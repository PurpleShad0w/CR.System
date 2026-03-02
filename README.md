# 📘 GTB / BACS Report Generation System


### Purpose of this README

This file is meant to be pasted at the start of any future conversation (human or LLM) so the full context of the project is immediately understood: architecture, data flow, constraints, and current implementation status.


## 1. What this project does

This project generates PowerPoint audit reports from OneNote field data, supporting two report families: BACS scoring reports (implemented) and État GTB reports (defined but not yet implemented).


## 2. Hard rules & constraints

Nothing under `archive/` is used. No design or implementation must rely on archived files.
OneNote extraction is external and handled via a Docker container (`onenote-exporter`). Python code consumes its outputs only.
Macro‑part 4 exists in configuration but is intentionally never generated (for all report types).
The system is designed as a compiler-style pipeline, not a conversational generator.


## 3. Report types

Report types are defined in `report_types.json`. The system currently distinguishes definition from implementation.

### 3.1 ✅ Implemented report type: BACS_SCORING

**Purpose**

Audit GTB systems under the Décret BACS / ISO 52120‑1, with scoring and target projection.

**Macro‑parts**

1. État des lieux GTB — generated, strictly descriptive
2. Scoring GTB actuel — generated, norm-based (ISO 52120‑1)
3. Scoring projeté — generated, inferred target (≥ classe B)
4. Mise en conformité & Bilan — defined but never generated

**Buckets**

 - `ETAT_DES_LIEUX_GTB` → macro‑part 1
 - `SCORING_ACTUEL` → macro‑part 2
 - `SCORING_PROJETE` → macro‑part 3
 - `CONFORMITE_BACS_BILAN` → macro‑part 4 (not generated)

### 3.2 🚧 Defined but NOT implemented: ETAT_GTB_AUDIT

**Purpose**

Pure GTB technical audit without BACS scoring (architecture, diagnostics, actions).

**Macro‑parts**

1. Généralités
2. État du système GTB
3. Diagnostics
4. Plan d’actions (never generated)

**Buckets** (already defined in config)

 - `SITE_CONTEXTE` → macro‑part 1
 - `ARCHI_GTB` → macro‑part 2
 - `COMMUNICATIONS` → macro‑part 2
 - `DIAGNOSTICS` → macro‑part 3

**⚠️ Status**

The system can detect this report type and plan buckets for it, but prompting, enforcement, and rendering are not implemented yet.


## 4. Two distinct pipelines

### 4.1 Generation pipeline (active, production)

Used to generate new audit reports.

```
OneNote export (Docker)
   ↓
process_onenote.py
   ↓
plan_generation.py
   ↓
generate_draft.py
   ↓
run_llm_jobs.py
   ↓
(enforcement / auto‑regen if enabled)
   ↓
render_report_pptx.py
```

Output: `Rapport_Audit.pptx`

### 4.2 Learning pipeline (active, support role)

Used to ingest existing audit reports and learn structure priors.

```
Existing reports (PDF / DOCX / PPTX)
   ↓
process_reports.py
   ↓
build_skeletons.py
   ↓
skeletons/*.json
```

This pipeline does not generate reports.


## 5. OneNote ingestion

### 5.1 onenote-exporter (Docker)

OneNote data is exported using a Docker container: `onenote-exporter`.
This repository does not implement OneNote extraction.

### 5.2 Expected exporter output

```
onenote-exporter/output/<notebook>/**/*.md
```

These Markdown files are consumed by `process_onenote.py`.


## 6. Script reference — generation pipeline

### 6.1 process_onenote.py

**Role**

Convert OneNote Markdown exports into structured JSON page packs.

**Input**

 - `onenote-exporter/output/<notebook>/**/*.md`

**Output**

 - `processed/<notebook>/pages/*.json`
 - `processed/<notebook>/assets/`
 - `manifest.json`, `errors.jsonl`

**Usage**

```
python process_onenote.py <notebook> --transcribe
```

### 6.2 plan_generation.py

**Role**

Build a per-case generation plan:

 - detect report type
 - select macro‑parts (1–3 only)
 - select and score buckets
 - route OneNote pages as evidence

**Input**

 - `report_types.json`
 - `processed/<notebook>/pages/*.json`
 - `skeletons/*.json` (priors only)

**Output**

 - `plans/<case_id>.json`

**Usage**

```
python plan_generation.py --config report_types.json --skeletons skeletons --onenote processed/<notebook>
```

### 6.3 generate_draft.py

**Role**

Translate the plan into LLM writing jobs.

**Responsibilities**

 - one prompt per bucket
 - inject macro‑part intent
 - encode forbidden / allowed semantics

**Input**

 - `plans/<case_id>.json`
 - `processed/<notebook>/pages/*.json`
 - `prompts/prompt_templates.json`

**Output**

 - `drafts/<case_id>/draft_bundle.json`
 - prompt text files for inspection

**Usage**

```
python generate_draft.py --plan plans/<case_id>.json --onenote processed/<notebook> --templates prompts/prompt_templates.json
```

### 6.4 run_llm_jobs.py

**Role**

Execute LLM calls and assemble macro‑parts.

**Input**

 - `drafts/<case_id>/draft_bundle.json`

**Output**

 - `generated_bundle.json`
 - `assembled_report.json` (macro‑parts 1–3)

**Usage**

```
python run_llm_jobs.py --bundle drafts/<case_id>/draft_bundle.json
```

### 6.5 llm_client.py

**Role**

HuggingFace chat client wrapper (LLaMA‑3.x).

**Environment variables**

 - `HF_TOKEN` (required)
 - `HF_MODEL` (optional, default provided)


### 6.6 render_report_pptx.py

**Role**

Render a final PowerPoint report from assembled_report.json and a PPTX template.

**Input**

 - `TEMPLATE_AUDIT_BUILD4USE.pptx`
 - `assembled_report.json`

**Output**

 - `Rapport_Audit.pptx`

**Usage**

```
python render_report_pptx.py --template TEMPLATE_AUDIT_BUILD4USE.pptx --assembled assembled_report.json --out Rapport_Audit.pptx
```


## 7. Script reference — learning tools

### 7.1 process_reports.py

**Role**

Normalize existing audit reports (PDF / DOCX / PPTX) into a structured corpus.

**Output**

 - `processed_reports/Rapports_d_audit/docs/`
 - `chunks/`
 - `assets/`
 - `manifest.json`

**Usage**

```
python process_reports.py --out processed_reports --extract-images
```

### 7.2 build_skeletons.py

**Role**

Learn canonical report skeletons (structure priors) from historical reports.

**Output**

 - `skeletons/GLOBAL.json`
 - `skeletons/BACS.json`
 - `skeletons/Etat_GTB_PlansActions.json`
 - `skeletons/Interventions_Avancement.json`

**Usage**

```
python build_skeletons.py --processed processed_reports/Rapports_d_audit --out skeletons
```

### 7.3 make_template_from_report.py

**Role**

Create a reusable PPTX template from a single report.

**Usage**

```
python make_template_from_report.py --in <report>.pptx --out TEMPLATE_AUDIT_BACS.pptx --keep-parts 1 2 3 --strip-images
```

### 7.4 derive_template_from_reports.py

**Role**

Derive a PPTX template from multiple reports by identifying common slide layouts.


## 8. Implemented vs planned

**Implemented end‑to‑end**

 - BACS report generation (macro‑parts 1–3)
 - OneNote → JSON normalization
 - Planning & routing
 - LLM draft generation
 - PPTX rendering
 - Corpus ingestion & skeleton learning

**Not implemented yet**

 - Full generation + enforcement + rendering for ETAT_GTB_AUDIT


## 9. Mental model recap

 - This is a compiler pipeline, not a chatbot.
 - Planning is deterministic.
 - LLMs are used only for phrasing.
 - Semantics are enforced outside the LLM.
 - BACS and État‑GTB are distinct report families.
 - État‑GTB is defined but intentionally not implemented yet.


## 10. Credits and Dependencies

- [hkevin01](https://github.com/hkevin01) for their OneNote Exporter.