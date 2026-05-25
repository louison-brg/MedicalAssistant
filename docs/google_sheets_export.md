# Export Google Sheets du benchmark

Le connecteur Google Drive de Codex peut rester "connecte" cote Codex meme si tu as supprime l'acces cote Google. Dans cette session, les actions `create` et `upload/import` renvoient encore un manque de permissions OAuth. Le classeur source est pret ici:

```bash
training/reports/medicalassistant_model_benchmarks.xlsx
```

## Option A - corriger le connecteur Codex

1. Ouvre la page Google des connexions tierces: https://myaccount.google.com/connections
2. Verifie aussi l'ancienne page des permissions: https://myaccount.google.com/permissions
3. Supprime les acces OpenAI / ChatGPT / Codex / Google Drive s'ils apparaissent.
4. Dans Codex ou ChatGPT, retourne dans les connecteurs/apps et reconnecte Google Drive.
5. Pendant l'ecran OAuth, accepte les permissions Drive de creation/import de fichiers, pas seulement la lecture.

Docs utiles:

- OpenAI - Google Drive connector setup: https://help.openai.com/en/articles/10948259-google-drive-synced-connectors-self-service-setup/
- OpenAI - connectors in ChatGPT: https://help.openai.com/en/articles/11487775-connectors-in-chatgpt
- Google - manage third-party account connections: https://support.google.com/accounts/answer/13533235
- Google - Drive API OAuth scopes: https://developers.google.com/workspace/drive/api/guides/api-specific-auth

## Option B - pipeline local independant de Codex

Ce pipeline contourne le connecteur Codex. Il utilise tes propres identifiants Google et importe le fichier `.xlsx` en Google Sheet natif.

1. Installe les dependances:

```bash
.venv/bin/pip install -r requirements-google.txt
```

2. Cree un OAuth client Google:

- Va sur https://console.cloud.google.com/apis/credentials
- Cree un projet si necessaire.
- Active Google Drive API.
- Cree un "OAuth client ID" de type "Desktop app".
- Telecharge le JSON et place-le ici:

```bash
.google/oauth_client.json
```

3. Lance l'export:

```bash
.venv/bin/python tools/export_benchmark_to_google_sheets.py \
  --xlsx training/reports/medicalassistant_model_benchmarks.xlsx \
  --title "MedicalAssistant Model Benchmarks - 2026-05-25"
```

Le script ouvrira l'autorisation Google dans le navigateur, gardera le token local dans `.google/token.json`, puis affichera l'URL du Google Sheet cree.

## Option service account

Si tu preferes un service account:

```bash
.venv/bin/python tools/export_benchmark_to_google_sheets.py \
  --service-account .google/service-account.json \
  --folder-id "DRIVE_FOLDER_ID" \
  --title "MedicalAssistant Model Benchmarks - 2026-05-25"
```

Le dossier Drive cible doit etre partage avec l'email du service account.
