"""API key authentication unit tests."""
import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from app.main import require_api_key, settings


def test_api_key_is_checked_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(settings, "app_env", "test")
    monkeypatch.setattr(settings, "api_key", SecretStr("secret-test-key"))
    with pytest.raises(HTTPException) as error:
        require_api_key("incorrect")
    assert error.value.status_code == 401
    require_api_key("secret-test-key")


def test_api_key_is_mandatory_in_production(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "app_env", "development")
    require_api_key(None)
    monkeypatch.setattr(settings, "app_env", "production")
    with pytest.raises(HTTPException) as error:
        require_api_key(None)
    assert error.value.status_code == 401
