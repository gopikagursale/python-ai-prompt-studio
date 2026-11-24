from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_prompt_studio.config import AppSettings


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(
        api_key="test-secret-key",
        model="test-model",
        request_timeout=12,
        max_retries=1,
        developer_instructions="Follow application rules.",
    )
