# UI locale (web) – Build4Use Report Generator

Cette mini-app Streamlit fournit une interface web locale pour lancer la pipeline `run_pipeline.py` via des boutons et des paramètres simples.

## Lancer
- Windows: `start.bat`
- macOS/Linux: `bash start.sh`

## Où déposer
Place le dossier `local_webapp_ui/` à la racine du projet (au même niveau que `run_pipeline.py`).

## Fonctionnement
- L'app appelle `run_pipeline.py` via `subprocess`.
- Les logs sont affichés en direct.
- À la fin, si le PPTX est trouvé, l'app propose un bouton de téléchargement.

## Personnalisation
- Édite `presets.json` pour ajouter des profils (ex: "BACS - Résidentiel", "Audit GTB", etc.).

## Sécurité / usage
- L'app tourne localement sur la machine (pas d'hébergement externe).
- Les chemins de fichiers restent sur le poste.
