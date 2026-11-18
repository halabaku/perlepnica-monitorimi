"""Njoftimet me email. Konfigurimi në .streamlit/secrets.toml (seksioni [smtp]) ose
në variablat e mjedisit SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_TO."""
import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

STATE = Path(__file__).parent / "data" / "state.json"


def smtp_config(secrets=None):
    s = dict(secrets or {})
    g = lambda k, d=None: s.get(k.lower().replace("smtp_", ""), os.environ.get(k, d))
    cfg = dict(host=g("SMTP_HOST"), port=int(g("SMTP_PORT", 587)), user=g("SMTP_USER"),
               password=g("SMTP_PASS"), to=s.get("to", os.environ.get("ALERT_TO")))
    return cfg if cfg["host"] and cfg["to"] else None


def send(cfg, subject, body):
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, cfg["user"] or "perlepnica@localhost", cfg["to"]
    msg.set_content(body)
    with smtplib.SMTP(cfg["host"], cfg["port"], timeout=20) as s:
        s.starttls()
        if cfg["user"]:
            s.login(cfg["user"], cfg["password"])
        s.send_message(msg)


def last_zone():
    return json.loads(STATE.read_text()).get("zone", 0) if STATE.exists() else 0


def save_zone(z):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps({"zone": int(z)}))
