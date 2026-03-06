
# Build4Use GTB / BACS Report Generation Pipeline

This repository implements a **section-centric, evidence-driven pipeline** for generating GTB / BACS audit reports (PPTX) from OneNote field data.

The pipeline has evolved from a page-based approach to a **OneNote section–level synthesis model**, which aligns with how real audit reports are produced by humans.

---

## Core Design Principles

### 1. The Report Unit Is a OneNote Section

A **single report corresponds to one OneNote section**, for example:

- `Oseraie – OSNY`
- `Clinique – Goussonville`

A OneNote *page* (e.g. “CTA 3 salle à manger”) is **never** treated as a report or slide on its own. Pages are field observations that must be **aggregated and synthesized**.

The canonical identifier (`case_id`) for a report is the **slug of the OneNote section name**:

```text
"Oseraie – OSNY" → oseraie_osny
```

All pipeline artifacts (plans, drafts, outputs) are keyed by this slug.

---

### 2. Pages → Section Synthesis → Report Text

The pipeline explicitly separates three conceptual layers:

1. **Field notes** (raw OneNote pages: photos, short captions, headings)
2. **Section-level synthesis** (inventory, clustering, recurring observations)
3. **Narrative report text** (paragraphs, inventories, diagnostics)

This avoids the common failure mode of asking an LLM to write a report directly from sparse page snippets.

---

## Pipeline Stages

### 0. OneNote Processing

**Script**: `process_onenote.py`

- Converts OneNote exports into structured JSON page packs
- Preserves:
  - section name
  - page title
  - blocks (`heading`, `paragraph`, `image`, `image_ocr`, `audio`)
  - asset references (images)

Output:
```
process/onenote/<notebook>/pages/*.json
```

---

### 1. Section-Level Aggregation (NEW)

**Script**: `aggregate_onenote_section.py`

This step aggregates **all pages belonging to one OneNote section** into a structured synthesis artifact.

What it produces:
- Equipment clustering by family (CTA, groupes froids, automates, etc.)
- Inventory counts per family
- Page index with titles and asset counts
- Explicit reminder that pages are not mapped 1:1 to slides

Output:
```
process/onenote_aggregates/<notebook>/<section_slug>.json
```

This artifact is **template-agnostic** and reused by downstream steps.

---

### 2. Planning

**Script**: `plan_generation.py`

Inputs:
- OneNote pages **filtered to the selected section only**
- Report type configuration
- Learned skeleton catalogs (structure priors)

Responsibilities:
- Detect report type (e.g. `BACS_SCORING`)
- Select macro-parts to generate (1–3 only)
- Select relevant section buckets
- Route relevant pages to each bucket

The generated plan explicitly records the `onenote_section` it applies to.

Output:
```
process/plans/<section_slug>.json
```

---

### 3. Draft Generation (Section-Aware Prompting)

**Script**: `generate_draft.py`

This step **changes the LLM prompting contract**.

Instead of:
> “Write from these page snippets”

The model is instructed:
> “Write the report section based on the synthesized view of the entire OneNote section.”

What is injected into each prompt:
- Section-level synthesis (inventory, families, page counts)
- Explicit instruction to:
  - reproduce inventories when evidence is list-like
  - synthesize across pages (not page-by-page paraphrase)
- Then, detailed per-page evidence excerpts

Outputs:
```
process/drafts/<section_slug>/
  ├─ draft_bundle.json
  ├─ prompts.txt
  └─ prompts/<bucket_id>.txt
```

---

### 4. LLM Execution (Multi-Step)

**Script**: `run_llm_jobs.py`

Default mode: `multistep`

Per section bucket:
1. **Facts extraction** (JSON)
2. **Writing from structured facts**
3. **Optional repair pass** (semantic / compliance fixes)

The output preserves:
- `facts_json`
- `draft_text`
- `final_text`

Backward compatibility is maintained (`generated_text` always populated).

Outputs:
```
process/drafts/<section_slug>/
  ├─ generated_bundle.json
  ├─ assembled_report.json
  └─ quality_report.json
```

---

### 5. Rendering (Pagination, Not Interpretation)

**Script**: `render_report_pptx.py`

Responsibilities:
- Paginate text into slides
- Split content by semantic headings (`####`)
- Never reinterpret meaning or structure

Important rule:
> The renderer decides pagination, not content.

Output:
```
output/reports/<section_slug>/Rapport_Audit.pptx
```

---

## End-to-End Example

### Example: OneNote Section “Oseraie – OSNY”

**Input (OneNote)**:
- Section: `Oseraie – OSNY`
- Pages:
  - `CTA 3 – Salle à manger`
  - `CTA 4`
  - `Groupe froid – Terrasse`
  - `Extracteurs`
  - `Local automates`

These pages contain mostly photos, short captions, and repeated equipment observations.

**Aggregation Output**:
```
process/onenote_aggregates/test/oseraie_osny.json
```
Contains:
- Inventory: 3 CTA, 1 groupe froid, multiple extracteurs
- Recurrent issues: missing identification, local screens not powered

**Plan**:
```
process/plans/oseraie_osny.json
```
Defines which buckets (e.g. État des lieux GTB, Architecture GTB) will be generated.

**Drafts & LLM Output**:
```
process/drafts/oseraie_osny/
  ├─ draft_bundle.json
  ├─ generated_bundle.json
  ├─ assembled_report.json
  └─ quality_report.json
```

**Final Report**:
```
output/reports/oseraie_osny/Rapport_Audit.pptx
```

The resulting slides contain:
- Paragraphs synthesizing observations across multiple pages
- Explicit equipment inventories
- No page-by-page narration

---

## Running the Pipeline

```bash
python run_pipeline.py   --notebook test   --onenote-section "Oseraie - OSNY"
```

---

## Key Rules (Non-Negotiable)

- ❌ OneNote pages are **not** slides
- ❌ Pages are **not** summarized independently
- ✅ All report text is synthesized at **section level**
- ✅ Inventories must be preserved and reproduced
- ✅ Templates control layout, not content logic

---

*This README documents the current canonical behavior of the pipeline. Any future template evolution must respect the section-level synthesis contract described above.*
