"""The API envelope every JSON endpoint answers with, and the engine's result tuple.

One shape for every response, success or failure, so the browser branches on
``status`` alone:

    {"status": true, "status_code": 200, "message": "", "data": {}}

``message`` is empty on success and carries a user-facing sentence on failure;
the browser prints it verbatim.

Flask is imported inside :func:`check_json_data` only, so the engine can import
:func:`make_function_result` from here without depending on Flask.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
WRONG_FORMAT_MESSAGE = "Wrong request format"

ApiResponse = dict[str, Any] | tuple[dict[str, Any], int]
FunctionResult = tuple[bool, str, dict[str, Any]]


def make_api_response(
    status: bool = True,
    message: str = "",
    data: dict[str, Any] | None = None,
    status_code: int = HTTP_OK,
    reflect_status: bool = False,
) -> ApiResponse:
    """Build the standard API envelope.

    Args:
        status: True on success.
        message: User-facing error message, empty on success.
        data: Result payload.
        status_code: HTTP status code, mirrored into the body.
        reflect_status: When True, return (body, status_code) so Flask sets the HTTP status.

    Returns:
        The response dict, or a (dict, int) tuple when reflect_status is True.
    """
    response = {
        "status": status,
        "status_code": status_code,
        "message": message,
        "data": data if data is not None else {},
    }

    return (response, status_code) if reflect_status else response


def make_function_result(result: bool, message: str, data: dict[str, Any]) -> FunctionResult:
    """Build the standard engine return tuple.

    Args:
        result: True on success.
        message: User-facing error message, empty on success.
        data: Result payload (on failure, may carry a ``reason`` the route maps to a status code).

    Returns:
        The (result, message, data) tuple.
    """
    return (result, message, data)


def check_json_data(view: Callable[..., ApiResponse]) -> Callable[..., ApiResponse]:
    """Parse the JSON request body and pass it as the view's first argument.

    Returns a 400 envelope when the body is not a JSON object. Endpoints reading
    ``request.form`` or ``request.files`` must not use this.

    Args:
        view: The view function to wrap.

    Returns:
        The wrapped view.
    """

    @wraps(view)
    def decorator(*args: Any, **kwargs: Any) -> ApiResponse:
        """Parse the body once, then call the view with it."""
        from flask import request

        json_data: Any = None
        try:
            json_data = request.get_json()
        except Exception:  # noqa: BLE001 - any parse failure is the same answer
            json_data = None

        response: ApiResponse
        if not isinstance(json_data, dict):
            response = make_api_response(
                status=False,
                message=WRONG_FORMAT_MESSAGE,
                data={},
                status_code=HTTP_BAD_REQUEST,
                reflect_status=True,
            )
        else:
            response = view(json_data, *args, **kwargs)

        return response

    return decorator
