from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.week5.api_logging import RequestLogger, build_log_record


class APIError(Exception):
    """Application-level error with explicit HTTP status."""

    def __init__(
        self,
        status_code: int,
        error: str,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error = error
        self.message = message
        self.details = details or {}


def _sanitize_json_value(value: Any) -> Any:
    """
    Recursively convert non-JSON-serializable values to strings.
    Handles Pydantic's ValueError objects in ctx.error, Exceptions, etc.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    return str(value)


def _sanitize_errors(errors: list[Any]) -> list[Any]:
    return [_sanitize_json_value(e) for e in errors[:5]]


def install_error_handlers(app: FastAPI, logger: RequestLogger) -> None:
    """
    Register 422, APIError (400/500/503), and generic 500 handlers.
    """

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        safe_errors = _sanitize_errors(exc.errors())

        record = build_log_record(
            request,
            status_code=422,
            error_type="RequestValidationError",
            extra={"errors": safe_errors},
        )
        logger.log(record)

        return JSONResponse(
            status_code=422,
            content={
                "request_id": record["request_id"],
                "error": "validation_error",
                "message": "Request body failed validation.",
                "details": {"errors": safe_errors},
            },
        )

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        safe_details = _sanitize_json_value(exc.details)

        record = build_log_record(
            request,
            status_code=exc.status_code,
            error_type=exc.error,
            extra={"details": safe_details},
        )
        logger.log(record)

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "request_id": record["request_id"],
                "error": exc.error,
                "message": exc.message,
                "details": safe_details,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        record = build_log_record(
            request,
            status_code=500,
            error_type=type(exc).__name__,
            extra={"message": str(exc)[:500]},
        )
        logger.log(record)

        return JSONResponse(
            status_code=500,
            content={
                "request_id": record["request_id"],
                "error": "internal_server_error",
                "message": "An unexpected error occurred.",
                "details": {"error_type": type(exc).__name__},
            },
        )
