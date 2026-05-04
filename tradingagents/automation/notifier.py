"""Phone notification helpers for the automation service."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from email.header import Header
from pathlib import Path
from typing import Iterable, Optional
from urllib import parse, request

logger = logging.getLogger(__name__)


class NtfyNotifier:
    """Thin wrapper around the ntfy publish API."""

    def __init__(self, config: dict):
        topic = str(config.get("ntfy_topic", "") or "").strip()
        self.enabled = bool(config.get("ntfy_enabled", False) and topic)
        self.server = str(config.get("ntfy_server", "https://ntfy.sh")).rstrip("/")
        self.topic = topic
        self.default_priority = str(config.get("ntfy_priority", "default"))
        raw_tags = config.get("ntfy_tags", [])
        if isinstance(raw_tags, str):
            raw_tags = [item.strip() for item in raw_tags.split(",") if item.strip()]
        self.default_tags = [str(item).strip() for item in raw_tags if str(item).strip()]
        click_url = str(config.get("ntfy_click_url", "") or "").strip()
        self.click_url = click_url or None
        self.state_path = (
            Path(config.get("results_dir", "./results"))
            / "notifications"
            / "ntfy_state.json"
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def send(
        self,
        title: str,
        message: str,
        *,
        priority: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        dedupe_key: Optional[str] = None,
        click: Optional[str] = None,
    ) -> bool:
        if not self.enabled:
            return False
        if dedupe_key and self._already_sent(dedupe_key):
            return False

        final_tags = [tag for tag in self.default_tags]
        if tags:
            final_tags.extend(str(tag).strip() for tag in tags if str(tag).strip())

        headers = {
            "Title": self._encode_header_value(title),
            "Priority": self._encode_header_value(str(priority or self.default_priority)),
        }
        if final_tags:
            headers["Tags"] = self._encode_header_value(",".join(dict.fromkeys(final_tags)))
        if click or self.click_url:
            headers["Click"] = self._encode_header_value(click or self.click_url)

        publish_url = f"{self.server}/{parse.quote(self.topic, safe='')}"
        payload = message.encode("utf-8")
        req = request.Request(
            publish_url,
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=10) as resp:
                resp.read()
            if dedupe_key:
                self._mark_sent(dedupe_key)
            logger.info("Sent ntfy notification to topic %s", self.topic)
            return True
        except Exception as exc:
            logger.warning("Failed to send ntfy notification: %s", exc)
            return False

    def _encode_header_value(self, value: str) -> str:
        try:
            value.encode("latin-1")
            return value
        except UnicodeEncodeError:
            return Header(value, "utf-8").encode()

    def _already_sent(self, key: str) -> bool:
        state = self._load_state()
        return key in state

    def _mark_sent(self, key: str):
        state = self._load_state()
        state[key] = datetime.now(timezone.utc).isoformat()
        if len(state) > 500:
            items = sorted(state.items(), key=lambda item: item[1], reverse=True)[:500]
            state = dict(items)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
