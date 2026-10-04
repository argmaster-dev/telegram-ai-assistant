"""Exception hierarchy for the OCR + Gemini pipeline.

Lives in its own module so ``gemini_client``, ``ocr_gemini``, ``ocr_router``
can import it without forming an import cycle.
"""


class OcrError(Exception):
    """Base class for all OCR / Gemini pipeline errors."""


class ConfigError(OcrError):
    """Wrong configuration: missing API key, invalid ``mode``, etc."""


class PrivacyStrictBlocked(OcrError):
    """Raised when ``PRIVACY_STRICT=true`` blocks a Gemini-bound call."""


class GeminiUnavailable(OcrError):
    """Raised on Gemini network errors, 5xx, 429, or malformed responses.

    NOTE: ``ocr_router`` deliberately does NOT silently fall back to local OCR
    when this fires in PUBLIC mode — the user explicitly chose PUBLIC for
    quality and must not silently get a degraded result.
    """
