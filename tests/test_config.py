"""Tests for configuration settings."""

from src.rag.config import Settings

# Models known to be retired/unavailable via the Anthropic API (404 not_found_error).
DEPRECATED_ANTHROPIC_MODELS = {"claude-3-5-haiku-20241022"}


class TestAnthropicModelSetting:
    """Tests for the anthropic_model configuration field."""

    def test_default_model_is_not_deprecated(self):
        """Default anthropic_model must not be a retired model id.

        Regression test: claude-3-5-haiku-20241022 was retired and requests
        using it fail with a 404 not_found_error at generation time.
        """
        settings = Settings()
        assert settings.anthropic_model not in DEPRECATED_ANTHROPIC_MODELS

    def test_default_model_value(self):
        """Default anthropic_model should be the current Haiku model."""
        settings = Settings()
        assert settings.anthropic_model == "claude-haiku-4-5-20251001"

    def test_model_overridable_via_env(self, monkeypatch):
        """anthropic_model should be overridable via ANTHROPIC_MODEL env var."""
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5")
        settings = Settings()
        assert settings.anthropic_model == "claude-sonnet-5"
