"""
ClusterHealth AI — Live Alerting
=================================
Sends real notifications to Slack, Discord, and/or Email when a device
crosses into high/medium risk. Wired into api.py so alerts fire the
moment a prediction crosses a threshold.

Configuration is via environment variables (see docker/.env.example):
  SLACK_WEBHOOK_URL     - Slack incoming webhook URL
  DISCORD_WEBHOOK_URL   - Discord webhook URL
  ALERT_EMAIL_FROM      - sender address (needs SMTP creds below)
  ALERT_EMAIL_TO        - comma-separated recipient list
  SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD

Any channel left unconfigured (empty env var) is silently skipped --
you don't need all three wired up for the demo to work.

De-duplication: an alert only fires once per device per risk-level
transition (e.g. low->high), not on every single API poll, so you don't
spam the channel every few seconds.
"""

import os
import smtplib
import ssl
import threading
from email.mime.text import MIMEText
import requests

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
ALERT_EMAIL_FROM = os.environ.get("ALERT_EMAIL_FROM", "")
ALERT_EMAIL_TO = os.environ.get("ALERT_EMAIL_TO", "")
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")

# in-memory dedup: {device_id: last_risk_level_alerted}
_last_alerted_level: dict[int, str] = {}
_alert_lock = threading.Lock()


def _format_message(device_id: int, risk_level: str, prob: float, top_reasons, action: str) -> str:
    reasons = ", ".join(f"{name} ({val:+.2f})" for name, val in top_reasons[:3])
    emoji = "🔴" if risk_level == "high" else "🟡"
    return (
        f"{emoji} ClusterHealth AI Alert — Device {device_id}\n"
        f"Risk level: {risk_level.upper()}  |  Failure probability: {prob*100:.1f}%\n"
        f"Top drivers: {reasons}\n"
        f"Action: {action}"
    )


def _send_slack(text: str):
    if not SLACK_WEBHOOK_URL:
        return
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=5).raise_for_status()
    except Exception as e:
        print(f"[alerts] Slack send failed: {e}")


def _send_discord(text: str):
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": text}, timeout=5).raise_for_status()
    except Exception as e:
        print(f"[alerts] Discord send failed: {e}")


def _send_email(subject: str, body: str):
    if not (SMTP_HOST and ALERT_EMAIL_FROM and ALERT_EMAIL_TO):
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = ALERT_EMAIL_FROM
        msg["To"] = ALERT_EMAIL_TO

        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls(context=context)
            if SMTP_USER:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(ALERT_EMAIL_FROM, ALERT_EMAIL_TO.split(","), msg.as_string())
    except Exception as e:
        print(f"[alerts] Email send failed: {e}")


def maybe_alert(device_id: int, risk_level: str, prob: float, top_reasons, action: str):
    """
    Call this after every prediction. Only actually sends a notification
    if the device's risk level has CHANGED since the last alert (dedup),
    and only for medium/high risk (not "low").
    """
    with _alert_lock:
        if risk_level == "low":
            _last_alerted_level.pop(device_id, None)
            return
        if _last_alerted_level.get(device_id) == risk_level:
            return  # already alerted for this risk level, don't spam
        _last_alerted_level[device_id] = risk_level
    text = _format_message(device_id, risk_level, prob, top_reasons, action)

    _send_slack(text)
    _send_discord(text)
    _send_email(f"[ClusterHealth AI] Device {device_id} — {risk_level.upper()} risk", text)

    print(f"[alerts] Fired for device {device_id} ({risk_level}): "
          f"slack={'on' if SLACK_WEBHOOK_URL else 'off'} "
          f"discord={'on' if DISCORD_WEBHOOK_URL else 'off'} "
          f"email={'on' if SMTP_HOST else 'off'}")
