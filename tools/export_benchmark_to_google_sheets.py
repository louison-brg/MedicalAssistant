#!/usr/bin/env python3
"""
Upload the benchmark workbook as a native Google Sheet.

This pipeline is intentionally independent from the Codex Google Drive
connector. It uses your own Google OAuth client or a service account.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
GOOGLE_SHEET_MIME = "application/vnd.google-apps.spreadsheet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export benchmark .xlsx to Google Sheets.")
    parser.add_argument(
        "--xlsx",
        default="training/reports/medicalassistant_model_benchmarks.xlsx",
        help="Path to the benchmark workbook to upload.",
    )
    parser.add_argument(
        "--title",
        default="MedicalAssistant Model Benchmarks",
        help="Google Sheet title.",
    )
    parser.add_argument(
        "--credentials",
        default=".google/oauth_client.json",
        help="OAuth client JSON from Google Cloud Console.",
    )
    parser.add_argument(
        "--token",
        default=".google/token.json",
        help="Cached OAuth token path. Ignored with --service-account.",
    )
    parser.add_argument(
        "--service-account",
        default=None,
        help="Optional service account JSON. If set, OAuth browser flow is skipped.",
    )
    parser.add_argument(
        "--folder-id",
        default=None,
        help="Optional Drive folder ID to place the new Google Sheet in.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Print auth URL instead of opening the browser automatically.",
    )
    return parser.parse_args()


def fail_missing_deps(exc: Exception) -> None:
    print("Missing Google export dependencies.", file=sys.stderr)
    print("Install them with:", file=sys.stderr)
    print("  .venv/bin/pip install -r requirements-google.txt", file=sys.stderr)
    print(f"Original import error: {exc}", file=sys.stderr)
    raise SystemExit(2)


def load_credentials(args: argparse.Namespace):
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google.oauth2 import service_account
        from google_auth_oauthlib.flow import InstalledAppFlow
    except Exception as exc:  # pragma: no cover - dependency guard
        fail_missing_deps(exc)

    if args.service_account:
        service_account_path = Path(args.service_account)
        if not service_account_path.exists():
            raise FileNotFoundError(f"Service account file not found: {service_account_path}")
        return service_account.Credentials.from_service_account_file(
            str(service_account_path),
            scopes=SCOPES,
        )

    token_path = Path(args.token)
    credentials_path = Path(args.credentials)
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    if not creds or not creds.valid:
        if not credentials_path.exists():
            raise FileNotFoundError(
                f"OAuth client file not found: {credentials_path}\n"
                "Create it in Google Cloud Console and save it there, or pass --credentials."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
        creds = flow.run_local_server(port=0, open_browser=not args.no_browser)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return creds


def upload_xlsx_as_google_sheet(creds: Any, xlsx: Path, title: str, folder_id: str | None) -> dict[str, Any]:
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except Exception as exc:  # pragma: no cover - dependency guard
        fail_missing_deps(exc)

    metadata: dict[str, Any] = {
        "name": title,
        "mimeType": GOOGLE_SHEET_MIME,
    }
    if folder_id:
        metadata["parents"] = [folder_id]

    media = MediaFileUpload(str(xlsx), mimetype=XLSX_MIME, resumable=False)
    service = build("drive", "v3", credentials=creds)
    return (
        service.files()
        .create(
            body=metadata,
            media_body=media,
            fields="id,name,mimeType,webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )


def main() -> int:
    args = parse_args()
    xlsx = Path(args.xlsx)
    if not xlsx.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx}")

    creds = load_credentials(args)
    result = upload_xlsx_as_google_sheet(creds, xlsx, args.title, args.folder_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("webViewLink"):
        print(f"\nGoogle Sheet: {result['webViewLink']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
