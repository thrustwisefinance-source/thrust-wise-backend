"""
Global exception handlers that ensure every error response uses the
ApiResponse envelope: { data: null, success: false, message: "..." }.

Without these, FastAPI returns raw {"detail": "..."} which breaks
the frontend's error parsing.
"""
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Call once from main.py after app creation."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "data": None,
                "success": False,
                "message": exc.detail
                if isinstance(exc.detail, str)
                else str(exc.detail),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Summarize validation errors without leaking internals
        errors = exc.errors()
        messages = []
        for err in errors:
            loc = " → ".join(str(part) for part in err.get("loc", []))
            messages.append(f"{loc}: {err.get('msg', 'invalid')}")
        return JSONResponse(
            status_code=422,
            content={
                "data": None,
                "success": False,
                "message": "; ".join(messages) if messages else "Validation error",
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception(
            "Unhandled exception on %s %s", request.method, request.url.path
        )
        return JSONResponse(
            status_code=500,
            content={
                "data": None,
                "success": False,
                "message": "Internal server error",
            },
        )
