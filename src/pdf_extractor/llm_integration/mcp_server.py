# -*- coding: utf-8 -*-
"""
Description: Implements an MCP (Model Context Protocol) server that acts as a
             bridge to the LiteLLM FastAPI service.

This server exposes the FastAPI service's batch processing capabilities as an
MCP tool, allowing MCP clients (like AI agents) to interact with the LiteLLM
service through the standardized MCP interface.

Core Libraries/Concepts:
------------------------
- modelcontextprotocol: Library for building MCP servers and clients.
  (Likely an internal library - no public URL)
- asyncio: For running the asynchronous server.
- httpx: For making asynchronous HTTP calls to the backend FastAPI service. # Changed from requests

Key Components:
--------------
- LiteLLMMcpServer: The main class implementing the MCP server logic.
- _handle_list_tools: Responds to MCP requests asking for available tools.
- _handle_call_tool: Handles requests to execute the 'litellm_batch_ask' tool
                     by proxying the request to the FastAPI endpoint.
- run: Starts the server using stdio transport.

Sample I/O (Conceptual - CallTool):
-----------------------------------
Input (MCP Request):
  {
    "mcp_version": "0.1",
    "request_id": "req-123",
    "type": "CallToolRequest",
    "params": {
      "name": "litellm_batch_ask",
      "arguments": {"questions": [...]} // FastAPI BatchRequest structure
    }
  }
Output (MCP Response):
  {
    "mcp_version": "0.1",
    "request_id": "req-123",
    "type": "CallToolResponse",
    "content": [{"type": "text", "text": "{...}"}] // JSON string from FastAPI BatchResponse
  }
"""

import asyncio
import json
import os
import sys
# import requests # Removed synchronous requests
import httpx # Added httpx for async requests
import httpx # Added httpx for async requests
from modelcontextprotocol.server import Server, StdioServerTransport
from modelcontextprotocol.types import (
    CallToolRequestSchema,
    CallToolResponseSchema,
    ErrorCode,
    ListToolsRequestSchema,
    ListToolsResponseSchema,
    McpError,
    TextContentSchema,
)
from loguru import logger # Added for validation logging

# Ensure the src directory is in the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Configuration ---
SERVER_NAME = "mcp-litellm-batch-py"
SERVER_VERSION = "0.1.0"
TOOL_NAME = "litellm_batch_ask"
# Use localhost as the MCP server runs alongside the FastAPI service (likely in the same container eventually, or on the host)
FASTAPI_ENDPOINT = os.environ.get("MCP_LITELLM_FASTAPI_ENDPOINT", "http://localhost:8000/ask")

# --- MCP Server Implementation ---

class LiteLLMMcpServer:
    """
    MCP Server that acts as a bridge to the LiteLLM FastAPI service.
    """
    def __init__(self):
        self.server = Server(
            {"name": SERVER_NAME, "version": SERVER_VERSION},
            {"capabilities": {"tools": {}}}, # Announce tool capability
        )
        self._setup_handlers()
        self.server.onerror = self._handle_error

    def _handle_error(self, error: McpError):
        """Logs MCP errors."""
        print(f"[MCP Error] Code: {error.code}, Message: {error.message}", file=sys.stderr)
        if error.data:
            print(f"  Data: {error.data}", file=sys.stderr)

    def _setup_handlers(self):
        """Sets up handlers for MCP requests."""
        self.server.setRequestHandler(ListToolsRequestSchema, self._handle_list_tools)
        self.server.setRequestHandler(CallToolRequestSchema, self._handle_call_tool)

    async def _handle_list_tools(self, request: ListToolsRequestSchema) -> ListToolsResponseSchema:
        """Handles requests to list available tools."""
        # request parameter is unused in this implementation, but kept for signature consistency
        return {
            "tools": [
                {
                    "name": TOOL_NAME,
                    "description": "Sends a batch of questions to the LiteLLM service for processing.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {"type": "object"}, # Keep it flexible, FastAPI validates
                                "description": "A list of question objects for batch processing."
                            }
                        },
                        "required": ["questions"],
                    },
                }
            ]
        }

    async def _handle_call_tool(self, request: CallToolRequestSchema) -> CallToolResponseSchema:
        """Handles requests to call a specific tool."""
        if request.params.name != TOOL_NAME:
            raise McpError(ErrorCode.MethodNotFound, f"Unknown tool: {request.params.name}")

        tool_args = request.params.arguments
        if not isinstance(tool_args, dict) or "questions" not in tool_args:
             raise McpError(ErrorCode.InvalidParams, f"Invalid arguments for {TOOL_NAME}. Expected JSON object with 'questions' key.")

        try:
            # Call the FastAPI endpoint asynchronously
            async with httpx.AsyncClient(timeout=120.0) as client: # Use httpx async client
                response = await client.post(FASTAPI_ENDPOINT, json=tool_args)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Return the successful response from FastAPI
            return {
                "content": [
                    {
                        "type": "text",
                        "text": response.text # Return the raw JSON response text
                    }
                ]
            }
        except httpx.RequestError as e: # Catch httpx specific request errors
            # Handle network/request errors
            error_message = f"Error calling FastAPI service at {FASTAPI_ENDPOINT}: {e}"
            print(error_message, file=sys.stderr)
            # Return error information via MCP
            return {
                 "content": [
                    {
                        "type": "text",
                        "text": error_message
                    }
                ],
                "isError": True
            }
        except Exception as e:
             # Handle unexpected errors
            error_message = f"Unexpected error processing tool call: {e}"
            print(error_message, file=sys.stderr)
            raise McpError(ErrorCode.InternalError, error_message)


    async def run(self):
        """Connects to the transport and runs the server."""
        transport = StdioServerTransport()
        await self.server.connect(transport)
        print(f"{SERVER_NAME} v{SERVER_VERSION} running on stdio, proxying to {FASTAPI_ENDPOINT}", file=sys.stderr)
        # Keep the server running indefinitely
        await asyncio.Event().wait()

# --- Standalone Validation Block ---

async def main_validation():
    """Performs basic validation checks on the MCP server setup."""
    logger.info("--- Running Standalone Validation for mcp_server.py ---")
    validation_passed = True
    errors = []

    try:
        server_instance = LiteLLMMcpServer()
        logger.debug("LiteLLMMcpServer instance created.")

        # 1. Check server object
        if not hasattr(server_instance, 'server') or not isinstance(server_instance.server, Server):
            errors.append("MCP Server object not found or not initialized correctly.")
            validation_passed = False
        else:
            logger.debug("Internal MCP Server object found.")

            # 2. Check handler registration
            registered_handlers = server_instance.server.requestHandlers
            if ListToolsRequestSchema not in registered_handlers:
                errors.append("Handler for ListToolsRequestSchema not registered.")
                validation_passed = False
            else:
                logger.debug("ListToolsRequestSchema handler registered.")
            if CallToolRequestSchema not in registered_handlers:
                errors.append("Handler for CallToolRequestSchema not registered.")
                validation_passed = False
            else:
                logger.debug("CallToolRequestSchema handler registered.")

        # 3. Check tool definition (optional but good)
        try:
            # Simulate a dummy request object (or None if handler allows)
            # Note: This assumes _handle_list_tools doesn't rely on request content
            list_tools_response = await server_instance._handle_list_tools(None) # type: ignore
            if not list_tools_response or 'tools' not in list_tools_response or not list_tools_response['tools']:
                 errors.append("Tool list is empty or invalid.")
                 validation_passed = False
            else:
                tool = list_tools_response['tools'][0]
                if tool.get('name') != TOOL_NAME:
                    errors.append(f"Tool name mismatch: Expected '{TOOL_NAME}', Got '{tool.get('name')}'")
                    validation_passed = False
                else:
                    logger.debug(f"Tool '{TOOL_NAME}' found in tool list.")
                # Could add more checks for description, inputSchema etc.

        except Exception as e:
            errors.append(f"Error checking tool definition via _handle_list_tools: {e}")
            validation_passed = False


    except Exception as e:
        errors.append(f"Failed to instantiate LiteLLMMcpServer: {e}")
        validation_passed = False

    # Report validation status
    exit_code = 0
    if validation_passed:
        logger.success("✅ Standalone validation passed: MCP Server instance created, handlers registered, tool definition basic check passed.")
        print("\n✅ VALIDATION COMPLETE - Basic MCP server setup verified.")
    else:
        for error in errors:
            logger.error(f"❌ {error}")
        print("\n❌ VALIDATION FAILED - MCP server setup verification failed.")
        exit_code = 1

    # Attempt to cancel lingering tasks before exiting
    try:
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if tasks:
            logger.debug(f"Cancelling {len(tasks)} lingering asyncio tasks...")
            for task in tasks:
                task.cancel()
            # Allow cancellations to propagate
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.warning(f"Error cancelling tasks: {e}")

    sys.exit(exit_code)


# --- Main Execution ---
if __name__ == "__main__":
    # server_instance = LiteLLMMcpServer() # Original run code moved to run() method
    try:
        # Run validation instead of the server
        asyncio.run(main_validation())
        # To run the actual server:
        # asyncio.run(server_instance.run())
    except KeyboardInterrupt:
        # This part is less relevant now as validation exits, but keep for potential future use
        print("\nShutting down MCP server validation...", file=sys.stderr)
    except Exception as e:
         logger.critical(f"Critical error running main_validation: {e}", exc_info=True)
         sys.exit(1)