"""Typed exceptions for the Covalence client."""

from __future__ import annotations

from typing import Any


class CovalenceError(Exception):
    """Base exception for all Covalence client errors."""

    def __init__(self, message: str, *, status_code: int | None = None, response_body: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, status_code={self.status_code!r})"


class CovalenceBadRequestError(CovalenceError):
    """HTTP 400 — invalid request body, missing required field, unknown enum value,
    or business-rule violation."""


class CovalenceNotFoundError(CovalenceError):
    """HTTP 404 — the requested resource does not exist."""


class CovalenceServerError(CovalenceError):
    """HTTP 500 — unexpected database or internal engine error."""


class CovalenceHTTPError(CovalenceError):
    """Unexpected HTTP error (status codes other than 400, 404, 500)."""


class CovalenceConnectionError(CovalenceError):
    """Network-level error reaching the Covalence engine."""
