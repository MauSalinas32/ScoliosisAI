"""
Central exception -> HTTP mapping
=================================
Domain exceptions (utils/exceptions.py) that are not caught inside an endpoint are converted here to a
consistent JSON error. Internal details of unexpected errors are logged, never sent to the client.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from utils.exceptions import (
    AccountLockedError,
    AuthenticationError,
    AuthorizationError,
    FileSizeExceededError,
    ImageValidationError,
    InferenceError,
    InvalidModelInputError,
    MedicalSystemError,
    ModelNotFoundError,
    ModelNotLoadedError,
)
from utils.logger import get_logger

logger = get_logger("ErrorHandlers")

# Most specific first: the first matching class decides the status code.
STATUS_BY_EXCEPTION = (
    (AccountLockedError, 429),      # must come before AuthenticationError
    (AuthenticationError, 401),     # includes OneTimeCodeError
    (AuthorizationError, 403),
    (FileSizeExceededError, 413),
    (ImageValidationError, 422),    # unsupported format, corrupted file, image quality
    (InvalidModelInputError, 422),
    (ModelNotLoadedError, 503),
    (ModelNotFoundError, 404),
    (InferenceError, 500),
    (MedicalSystemError, 500),
)


def status_for(exc: Exception) -> int:
    for exc_type, status in STATUS_BY_EXCEPTION:
        if isinstance(exc, exc_type):
            return status
    return 500


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(MedicalSystemError)
    async def domain_error_handler(request: Request, exc: MedicalSystemError):
        status = status_for(exc)
        if status >= 500:
            logger.error(f"{type(exc).__name__} on {request.url.path}: {exc}")
            return JSONResponse(status_code=status, content={"detail": "Internal error. See the server log."})
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled {type(exc).__name__} on {request.url.path}")
        return JSONResponse(status_code=500, content={"detail": "Internal error. See the server log."})
