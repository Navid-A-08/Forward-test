"""Send the latest weekly report to Telegram.

Usage: TELEGRAM_TOKEN=... TELEGRAM_CHAT_ID=... python send_telegram.py [extra line]
The token is never stored in this repository.
"""

import html
import os
import sys
from pathlib import Path

import requests

token = os.environ["TELEGRAM_TOKEN"]
chat = os.environ["TELEGRAM_CHAT_ID"]
body = (Path(__file__).resolve().parent / "data" / "last_report.txt").read_text(encoding="utf-8")
extra = " ".join(sys.argv[1:]).strip()

text = f"<pre>{html.escape(body)}</pre>"
if extra:
    text += "\n" + html.escape(extra)

r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                  data={"chat_id": chat, "text": text[:4096], "parse_mode": "HTML",
                        "disable_web_page_preview": "true"}, timeout=30)
print(r.status_code, r.json().get("ok"))
sys.exit(0 if r.ok and r.json().get("ok") else 1)
