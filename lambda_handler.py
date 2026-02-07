"""
AWS Lambda handler for FastAPI application.

Supports two invocation sources with automatic detection:
- Lambda Function URL: No stage prefix, root_path is empty
- API Gateway (HTTP API v2): Stage prefix /prod, root_path is /prod

Mangum handles root_path via ASGI scope — no need to mutate app.root_path at runtime.
"""

import os
import logging
from mangum import Mangum

logger = logging.getLogger("rag_app.lambda_handler")

# --- Constants ---
API_GATEWAY_BASE_PATH = "/prod"
FUNCTION_URL_STAGE = "$default"

# Create /tmp directories required by Lambda runtime
os.makedirs("/tmp/uploads", exist_ok=True)
os.makedirs("/tmp/cached_chunks", exist_ok=True)

# Import app after directory creation (app uses /tmp paths in Lambda)
from app.main import app, initialize_services

# --- Mangum Handlers ---
# API Gateway handler: strips /prod prefix from path, sets ASGI root_path="/prod"
_api_gateway_handler = Mangum(
    app, lifespan="off", api_gateway_base_path=API_GATEWAY_BASE_PATH
)
# Function URL handler: no prefix stripping, ASGI root_path="" (empty)
_function_url_handler = Mangum(app, lifespan="off")

# Flag to track lazy service initialization
_services_initialized = False


def _is_function_url_event(event):
    """
    Detect if the Lambda event originates from a Function URL vs API Gateway.

    Function URLs always set requestContext.stage to "$default".
    API Gateway uses custom stage names like "prod", "staging", etc.

    Args:
        event: Lambda event payload (v2 format)

    Returns:
        bool: True if event is from a Function URL, False for API Gateway
    """
    request_context = event.get("requestContext", {})
    stage = request_context.get("stage", "")
    return not stage or stage == FUNCTION_URL_STAGE


def handler(event, context):
    """
    Lambda entry point with lazy initialization and smart event routing.

    On first invocation, initializes all application services (embedding, vector DB,
    RAG, SQL, cache). Then detects whether the request came from a Lambda Function URL
    or API Gateway and routes to the appropriate Mangum handler.

    Args:
        event: Lambda event payload (v2 format from Function URL or HTTP API)
        context: Lambda context object with runtime metadata

    Returns:
        dict: HTTP response for the Lambda runtime
    """
    global _services_initialized

    # Initialize services on first invocation (not at import time to avoid init timeout)
    if not _services_initialized:
        logger.info("First invocation — initializing services...")
        initialize_services()
        _services_initialized = True

    # Route to the appropriate Mangum handler based on event source
    # Mangum sets ASGI scope root_path from api_gateway_base_path automatically
    if _is_function_url_event(event):
        return _function_url_handler(event, context)

    return _api_gateway_handler(event, context)
