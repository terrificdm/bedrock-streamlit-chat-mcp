"""
Tool Call Handler Module

This module provides data structures and functions for handling tool calls
in the MCP Client integration with AWS Bedrock Converse API.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any
import asyncio

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call request from the model"""
    tool_use_id: str
    tool_name: str
    tool_input: Dict[str, Any]


@dataclass
class ToolResult:
    """Represents the result of a tool execution"""
    tool_use_id: str
    content: List[Dict[str, Any]]
    status: str = "success"  # "success" or "error"


def extract_tool_calls_from_message(message_content: List[Dict]) -> List[ToolCall]:
    """
    Extract tool call requests from an assistant message.
    
    Args:
        message_content: The content array from an assistant message
        
    Returns:
        List of ToolCall objects extracted from the message
        
    Example message_content:
        [
            {
                "toolUse": {
                    "toolUseId": "tooluse_xxx",
                    "name": "read_file",
                    "input": {"path": "/tmp/file.txt"}
                }
            }
        ]
    """
    tool_calls = []
    
    for content_block in message_content:
        if "toolUse" in content_block:
            tool_use = content_block["toolUse"]
            tool_call = ToolCall(
                tool_use_id=tool_use["toolUseId"],
                tool_name=tool_use["name"],
                tool_input=tool_use.get("input", {})
            )
            tool_calls.append(tool_call)
            logger.info(f"Extracted tool call: {tool_call.tool_name} (ID: {tool_call.tool_use_id})")
    
    return tool_calls


async def execute_tool_calls(
    tool_calls: List[ToolCall],
    mcp_manager
) -> List[ToolResult]:
    """
    Execute multiple tool calls, potentially in parallel.
    
    Args:
        tool_calls: List of ToolCall objects to execute
        mcp_manager: MCPClientManager instance for tool execution
        
    Returns:
        List of ToolResult objects with execution results
    """
    if not tool_calls:
        logger.warning("Tool Handler: No tool calls to execute")
        return []
    
    logger.info(f"Tool Handler: Executing {len(tool_calls)} tool call(s)")
    
    # Log all tool calls
    for i, tool_call in enumerate(tool_calls):
        logger.info(f"Tool Handler: [{i+1}/{len(tool_calls)}] {tool_call.tool_name} (ID: {tool_call.tool_use_id})")
    
    try:
        # Create async tasks for all tool calls
        tasks = [
            _execute_single_tool(tool_call, mcp_manager)
            for tool_call in tool_calls
        ]
        
        # Execute all tools in parallel using asyncio.gather
        # return_exceptions=True ensures individual failures don't fail the entire batch
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle any exceptions
        tool_results = []
        success_count = 0
        error_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exception as error result
                tool_call = tool_calls[i]
                error_message = f"Unexpected exception: {str(result)}"
                logger.error(f"Tool Handler: Tool '{tool_call.tool_name}' raised exception: {error_message}", exc_info=result)
                tool_results.append(
                    ToolResult(
                        tool_use_id=tool_call.tool_use_id,
                        content=[{"text": error_message}],
                        status="error"
                    )
                )
                error_count += 1
            else:
                tool_results.append(result)
                if result.status == "success":
                    success_count += 1
                else:
                    error_count += 1
        
        # Log summary
        logger.info(f"Tool Handler: Execution complete - Success: {success_count}, Errors: {error_count}")
        
        return tool_results
        
    except Exception as e:
        # Handle catastrophic failure in batch execution
        logger.error(f"Tool Handler: Critical error during batch tool execution: {e}", exc_info=True)
        
        # Return error results for all tool calls
        error_results = []
        for tool_call in tool_calls:
            error_results.append(
                ToolResult(
                    tool_use_id=tool_call.tool_use_id,
                    content=[{"text": f"Batch execution failed: {str(e)}"}],
                    status="error"
                )
            )
        return error_results


async def _execute_single_tool(
    tool_call: ToolCall,
    mcp_manager
) -> ToolResult:
    """
    Execute a single tool call and return the result.
    
    Args:
        tool_call: ToolCall object to execute
        mcp_manager: MCPClientManager instance
        
    Returns:
        ToolResult with execution result or error
    """
    try:
        logger.info(f"Tool Handler: Executing tool '{tool_call.tool_name}'")
        logger.debug(f"Tool Handler: Input parameters: {tool_call.tool_input}")
        
        # Validate tool input
        if not isinstance(tool_call.tool_input, dict):
            error_message = f"Invalid tool input format: expected dict, got {type(tool_call.tool_input).__name__}"
            logger.error(f"Tool Handler: {error_message}")
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                content=[{"text": error_message}],
                status="error"
            )
        
        # Execute the tool via MCP manager
        result = await mcp_manager.execute_tool(
            tool_name=tool_call.tool_name,
            arguments=tool_call.tool_input
        )
        
        # Check if MCP manager returned an error
        if isinstance(result, dict) and result.get("status") == "error":
            error_message = result.get("content", "Unknown error")
            logger.error(f"Tool Handler: Tool '{tool_call.tool_name}' returned error: {error_message}")
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                content=[{"text": str(error_message)}],
                status="error"
            )
        
        # Format successful result
        # MCP returns result in various formats, normalize to content array
        result_content = result.get("content") if isinstance(result, dict) else result
        
        if isinstance(result_content, dict):
            content = [{"json": result_content}]
        elif isinstance(result_content, str):
            content = [{"text": result_content}]
        elif isinstance(result_content, list):
            # Already in content format
            content = result_content
        else:
            # Fallback: convert to string
            content = [{"text": str(result_content)}]
        
        logger.info(f"Tool Handler: Tool '{tool_call.tool_name}' executed successfully")
        logger.debug(f"Tool Handler: Result content: {content}")
        
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            content=content,
            status="success"
        )
        
    except TimeoutError as e:
        error_message = f"Tool execution timed out: {str(e)}"
        logger.error(f"Tool Handler: Tool '{tool_call.tool_name}' timed out: {error_message}")
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            content=[{"text": error_message}],
            status="error"
        )
    except ConnectionError as e:
        error_message = f"Connection error: {str(e)}"
        logger.error(f"Tool Handler: Connection error for tool '{tool_call.tool_name}': {error_message}")
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            content=[{"text": error_message}],
            status="error"
        )
    except ValueError as e:
        error_message = f"Invalid parameters: {str(e)}"
        logger.error(f"Tool Handler: Invalid parameters for tool '{tool_call.tool_name}': {error_message}")
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            content=[{"text": error_message}],
            status="error"
        )
    except Exception as e:
        # Catch any unexpected exception during tool execution
        error_message = f"Unexpected error: {str(e)}"
        logger.error(f"Tool Handler: Unexpected error executing tool '{tool_call.tool_name}': {error_message}", exc_info=True)
        
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            content=[{"text": error_message}],
            status="error"
        )


def format_tool_results_for_bedrock(tool_results: List[ToolResult]) -> Dict[str, Any]:
    """
    Format tool results as a user message for Bedrock Converse API.
    
    Args:
        tool_results: List of ToolResult objects
        
    Returns:
        Dictionary representing a user message with toolResult blocks
        
    Example output:
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tooluse_xxx",
                        "content": [{"json": {"result": "data"}}],
                        "status": "success"
                    }
                }
            ]
        }
    """
    content_blocks = []
    
    for tool_result in tool_results:
        content_blocks.append({
            "toolResult": {
                "toolUseId": tool_result.tool_use_id,
                "content": tool_result.content,
                "status": tool_result.status
            }
        })
    
    message = {
        "role": "user",
        "content": content_blocks
    }
    
    logger.info(f"Formatted {len(tool_results)} tool result(s) for Bedrock")
    
    return message


def create_tool_display_message(tool_call: ToolCall) -> str:
    """
    Create a user-friendly display message for tool usage in the UI.
    
    Args:
        tool_call: ToolCall object to display
        
    Returns:
        Formatted string for UI display
    """
    return f"🔧 Using tool: **{tool_call.tool_name}**"


def create_tool_result_display_message(tool_result: ToolResult) -> str:
    """
    Create a user-friendly display message for tool results in the UI.
    
    Args:
        tool_result: ToolResult object to display
        
    Returns:
        Formatted string for UI display
    """
    if tool_result.status == "success":
        return "✅ Tool executed successfully"
    else:
        return "❌ Tool execution failed"
