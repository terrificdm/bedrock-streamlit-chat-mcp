"""
Synchronous version of stream_conversation_with_tools for Streamlit compatibility
"""
import json
import logging
import streamlit as st
from typing import Optional, Dict
from botocore.exceptions import ClientError

from asyncio_loop_manager import get_or_create_event_loop_thread

logger = logging.getLogger(__name__)


def stream_conversation_with_tools_sync(
    bedrock_runtime,
    config,
    model_info,
    mcp_manager,
    tool_config: Optional[Dict] = None
):
    """
    Synchronous streaming function that handles multi-turn tool calling.
    
    This is a sync version compatible with Streamlit's execution model.
    """
    from tool_handler import (
        extract_tool_calls_from_message,
        execute_tool_calls,
        format_tool_results_for_bedrock,
        ToolResult
    )
    
    max_tool_rounds = 5
    current_round = 0
    
    # Build inference config
    inference_config = {"maxTokens": config.max_tokens}
    additional_model_fields = {}

    if config.enable_reasoning:
        if model_info.model_family == "claude":
            additional_model_fields = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": config.budget_tokens
                }
            }
        elif model_info.model_family == "qwen":
            additional_model_fields = {"reasoning_config": "high"}
    else:
        if config.temperature is not None:
            inference_config["temperature"] = config.temperature
        if config.top_p is not None and model_info.supports_top_p:
            inference_config["topP"] = config.top_p
        if model_info.model_family == "nova":
            additional_model_fields = {"inferenceConfig": {"top_k": config.top_k}} 
        elif model_info.supports_top_k:
            additional_model_fields = {"top_k": config.top_k}
    
    while current_round < max_tool_rounds:
        try:
            # Prepare API call parameters
            api_params = {
                "modelId": config.model_id,
                "messages": config.messages,
                "system": [{"text": config.system_message}],
                "inferenceConfig": inference_config,
                "additionalModelRequestFields": additional_model_fields
            }
            
            # Add tool config if available and model supports tools
            if tool_config and model_info.supports_tools:
                try:
                    api_params["toolConfig"] = tool_config
                    logger.debug(f"MCP: Added {len(tool_config.get('tools', []))} tools to Bedrock API call")
                except Exception as e:
                    logger.error(f"MCP: Error adding tool config to API params: {e}")
                    logger.warning("MCP: Continuing without tool support due to config error")
            elif tool_config and not model_info.supports_tools:
                logger.info(f"MCP: Model {config.model_id} does not support tools, skipping tool config")
            
            # Call Bedrock Converse Stream API
            response = bedrock_runtime.converse_stream(**api_params)
            
            # Process streaming response
            in_reasoning_block = False
            actual_response_text = ""
            reasoning_response_text = ""
            reasoning_redacted_content = b''
            signature_response_text = ""
            assistant_content = []
            stop_reason = None
            tool_use_blocks = {}
            
            for chunk in response["stream"]:
                # Extract stop reason
                if "messageStop" in chunk:
                    stop_reason = chunk["messageStop"].get("stopReason")
                    logger.info(f"Stream stopped with reason: {stop_reason}")
                
                # Handle content block start (for tool use)
                if "contentBlockStart" in chunk:
                    block_start = chunk["contentBlockStart"]
                    content_block_index = block_start.get("contentBlockIndex", 0)
                    
                    if "toolUse" in block_start.get("start", {}):
                        tool_use = block_start["start"]["toolUse"]
                        tool_use_blocks[content_block_index] = {
                            "toolUseId": tool_use.get("toolUseId", ""),
                            "name": tool_use.get("name", ""),
                            "input": ""
                        }
                
                # Handle content block delta
                if "contentBlockDelta" in chunk:
                    delta = chunk["contentBlockDelta"]["delta"]
                    content_block_index = chunk["contentBlockDelta"].get("contentBlockIndex", 0)
                    
                    # Handle reasoning content
                    if "reasoningContent" in delta:
                        if "text" in delta["reasoningContent"]:
                            reasoning_text = delta["reasoningContent"]["text"]
                            reasoning_response_text += reasoning_text
                            if not in_reasoning_block:
                                yield "----------------\n"
                                in_reasoning_block = True
                            yield reasoning_text
                        if "redactedContent" in delta["reasoningContent"]:
                            reasoning_redacted_content += delta["reasoningContent"]["redactedContent"]
                        if "signature" in delta["reasoningContent"]:
                            signature_response_text += delta["reasoningContent"]["signature"]
                    
                    # Handle text content
                    elif "text" in delta:
                        text = delta["text"]
                        actual_response_text += text
                        if in_reasoning_block:
                            yield "\n\n----------------\n"
                            in_reasoning_block = False
                        yield text
                    
                    # Handle tool use input
                    elif "toolUse" in delta:
                        if content_block_index in tool_use_blocks:
                            tool_input_chunk = delta["toolUse"].get("input", "")
                            tool_use_blocks[content_block_index]["input"] += tool_input_chunk
            
            # Build assistant content
            if reasoning_response_text and signature_response_text:
                assistant_content.append({
                    "reasoningContent": {
                        "reasoningText": {
                            "text": reasoning_response_text,
                            "signature": signature_response_text
                        }
                    }
                })
            if reasoning_redacted_content:
                assistant_content.append({
                    "reasoningContent": {
                        "redactedContent": reasoning_redacted_content
                    }
                })
            if actual_response_text:
                assistant_content.append({"text": actual_response_text})
            
            # Add tool use blocks
            for tool_use_data in tool_use_blocks.values():
                try:
                    tool_input = json.loads(tool_use_data["input"]) if tool_use_data["input"] else {}
                except json.JSONDecodeError:
                    tool_input = {}
                
                assistant_content.append({
                    "toolUse": {
                        "toolUseId": tool_use_data["toolUseId"],
                        "name": tool_use_data["name"],
                        "input": tool_input
                    }
                })
            
            # Add assistant message to history
            assistant_message = {
                "role": "assistant",
                "content": assistant_content
            }
            st.session_state.model_messages.append(assistant_message)
            
            # Handle tool use
            if stop_reason == "tool_use":
                logger.info(f"MCP: Tool use requested, extracting and executing tools (round {current_round + 1})")
                
                # Extract tool calls
                tool_calls = extract_tool_calls_from_message(assistant_content)
                
                if not tool_calls:
                    logger.warning("MCP: Stop reason was 'tool_use' but no tool calls found")
                    break
                
                # Display tool usage (tool name only, no arguments)
                yield "\n\n"
                for tool_call in tool_calls:
                    tool_name = tool_call.tool_name
                    yield f"🔧 Using tool: {tool_name} "
                
                # Execute tools using dedicated event loop thread
                try:
                    logger.info(f"MCP: Executing {len(tool_calls)} tool(s) using event loop thread")
                    
                    # Get the event loop thread from session state
                    loop_thread = get_or_create_event_loop_thread()
                    
                    # Execute tool calls in the dedicated event loop
                    tool_results = loop_thread.run_coroutine(execute_tool_calls(tool_calls, mcp_manager))
                    logger.info(f"MCP: Tool execution completed")
                    
                    # Display tool execution status (success/failure only, no result content)
                    for tool_result in tool_results:
                        if tool_result.status == "error":
                            yield "❌ Tool execution failed "
                        else:
                            yield "✅ Tool executed successfully "
                    
                except Exception as e:
                    logger.error(f"MCP: Critical error executing tool calls: {e}", exc_info=True)
                    yield "❌ Tool execution failed "
                    tool_results = []
                    for tool_call in tool_calls:
                        tool_results.append(
                            ToolResult(
                                tool_use_id=tool_call.tool_use_id,
                                content=[{"text": f"Tool execution system error: {str(e)}"}],
                                status="error"
                            )
                        )
                
                # Format tool results for Bedrock
                try:
                    tool_result_message = format_tool_results_for_bedrock(tool_results)
                except Exception as e:
                    logger.error(f"MCP: Error formatting tool results: {e}", exc_info=True)
                    yield f"\n\n⚠️ Error formatting tool results: {str(e)}\n"
                    break
                
                # Add tool results to model messages (for API)
                st.session_state.model_messages.append(tool_result_message)
                
                # Note: Do NOT add to display_messages here
                # Tool interaction text will be collected and added with final response
                
                # Continue conversation
                current_round += 1
                continue
                
            elif stop_reason == "end_turn" or stop_reason == "max_tokens":
                logger.info(f"Conversation completed with stop reason: {stop_reason}")
                break
            else:
                logger.info(f"Conversation stopped with reason: {stop_reason}")
                break
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            logger.error(f"AWS ClientError in conversation: {error_code} - {error_message}", exc_info=True)
            
            # Handle specific error types
            if error_code == 'ServiceUnavailableException':
                yield f"\n\n⚠️ AWS Bedrock服务暂时不可用，请稍后重试。\n"
                yield f"错误详情: {error_message}\n"
            elif error_code == 'ThrottlingException':
                yield f"\n\n⚠️ 请求频率过高，请稍后重试。\n"
            else:
                yield f"\n\n⚠️ AWS错误: {error_message}\n"
            break
            
        except Exception as e:
            logger.error(f"Error in conversation: {e}", exc_info=True)
            yield f"\n\n⚠️ 错误: {str(e)}\n"
            break
    
    if current_round >= max_tool_rounds:
        logger.warning(f"MCP: Reached maximum tool rounds ({max_tool_rounds})")
        yield f"\n\n⚠️ Reached maximum tool interaction rounds ({max_tool_rounds})\n"
