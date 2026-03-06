# 📘 GTB / BACS Report Generation System

> **But du README**
> Ce README est le point d’entrée du projet (à coller au début d’une future conversation) : architecture, flux de données, règles non négociables, et état d’implémentation.

## 1) Résumé (1 phrase)
Ce projet génère des **rapports d’audit PowerPoint (PPTX)** à partir de **données terrain OneNote**, via un pipeline type « compilateur » (ingestion → synthèse section → planification → rédaction LLM → assemblage → rendu PPTX).

---

## 2) Règles non négociables (hard constraints)
- **`archive/` n’est jamais utilisé** : aucune logique ne doit dépendre de fichiers archivés.
- **Extraction OneNote externe** : l’export OneNote est réalisé par un outil Docker (onenote-exporter). Le code Python consomme uniquement ses sorties.
- **Macro‑partie 4** : elle existe dans la configuration, mais elle est **intentionnellement jamais générée** (tous types de rapports).
- **Pages OneNote ≠ slides** : une page ne doit pas être résumée isolément comme une slide. Le rapport est synthétisé au niveau **section OneNote**.
- **Renderer PPTX = pagination** : il découpe/pagine du texte, il **n’interprète pas** le sens.

---

## 3) Concepts clés
### 3.1 Unité de rapport = une section OneNote
Un rapport correspond à **une section OneNote**, pas à une page. Le `case_id` est le **slug** du nom de section.

### 3.2 Trois couches de données
1) **Notes terrain** (pages OneNote brutes : titres, paragraphes, images, OCR, audio)
2) **Synthèse de section** (inventaires, familles d’équipements, index des pages)
3) **Texte du rapport** (macro-parties, sections, diagnostics)

---

## 4) Types de rapports (config)
Les types de rapports et leurs buckets sont définis dans `input/config/report_types.json`.

### 4.1 ✅ Implémenté : `BACS_SCORING`
Audit GTB sous **Décret BACS / ISO 52120‑1**, avec scoring actuel et projection.

Macro‑parties :
1. **État des lieux GTB** — générée
2. **Scoring GTB actuel** — générée
3. **Scoring projeté** — générée
4. **Mise en conformité & Bilan** — définie mais **jamais générée**

Buckets :
- `ETAT_DES_LIEUX_GTB` → macro‑partie 1
- `SCORING_ACTUEL` → macro‑partie 2
- `SCORING_PROJETE` → macro‑partie 3
- `CONFORMITE_BACS_BILAN` → macro‑partie 4 (non générée)

### 4.2 🚧 Défini mais pas end‑to‑end : `ETAT_GTB_AUDIT`
Audit technique (généralités, architecture, diagnostics). Planification/détection possible, mais le flux complet (enforcement / rendu dédié) n’est pas finalisé.

---

## 5) Pipeline de génération (end‑to‑end)
Entrypoint : `run_pipeline.py`.

```text
OneNote export (Docker)
  ↓
process_onenote.py
  ↓
aggregate_onenote_section.py      (synthèse section)
  ↓
plan_generation.py                (détection type + routing)
  ↓
generate_draft.py                 (jobs LLM par bucket)
  ↓
run_llm_jobs.py                   (LLM + assemblage + qualité)
  ↓
render_report_pptx.py             (pagination only)
  ↓
output/reports/<case_id>/Rapport_Audit.pptx
```

---

## 6) Scripts (référence rapide)
- `process_onenote.py` : transforme l’export Markdown en JSON structurés (pages + assets).
- `aggregate_onenote_section.py` : agrège toutes les pages d’une section en une synthèse réutilisable.
- `plan_generation.py` : choisit type/macro‑parties/buckets et route les preuves.
- `generate_draft.py` : produit `draft_bundle.json` (prompts bucket‑centrés).
- `run_llm_jobs.py` : exécute le LLM (single ou multistep), assemble `assembled_report.json`, calcule `quality_report.json`.
- `render_report_pptx.py` : rend le PPTX final à partir du template + `assembled_report.json`.
- `llm_client.py` : client HuggingFace Router (LLaMA), via `HF_TOKEN` / `HF_MODEL`.

---

## 7) Extension : Tableau 6 ISO 52120‑1 → scoring → slides (Partie 2)
### 7.1 Fichier de règles
Placer le JSON ici :
- `input/rules/bacs_table6_rules_structured_clean.json`

### 7.2 Principe
Pour `BACS_SCORING`, `run_llm_jobs.py` peut :
1) inférer un niveau (0..n ou null) pour chaque `rule_id` à partir de la synthèse OneNote,
2) calculer **déterministiquement** les classes A/B/C/D par groupe,
3) optionnellement demander à LLaMA de **mettre en forme** la macro‑partie 2 en “slides” (Markdown `####` + puces) pour le rendu PPTX.

### 7.3 Flags
- `--section_aggregate <path>`
- `--bacs_rules input/rules/bacs_table6_rules_structured_clean.json`
- `--bacs_building_scope "Résidentiel" | "Non résidentiel"`
- `--bacs_part2_slides` (active la génération slide‑ready en macro‑partie 2)

---

## 8) Exécution
### 8.1 Pipeline complet
```bash
python run_pipeline.py --notebook <NB> --onenote-section "<SECTION>"
```

### 8.2 Pipeline complet avec slides scoring (Partie 2)
```bash
python run_pipeline.py --notebook <NB> --onenote-section "<SECTION>" --bacs-part2-slides
```

---

## 9) Sous-composant : OneNote Exporter (Docker)
Le projet utilise un exporter Docker pour extraire OneNote via Microsoft Graph API.
La documentation complète de ce sous-composant est conservée dans :
- `input/onenote-exporter/README.md`

---

## 10) Aide‑mémoire (fast reload)
- Unité = **section OneNote**
- Pages ≠ slides
- Macro‑partie 4 jamais générée
- Renderer = pagination only
- `BACS_SCORING` implémenté
- Tableau 6 (JSON) possible pour scoring + slides Partie 2
