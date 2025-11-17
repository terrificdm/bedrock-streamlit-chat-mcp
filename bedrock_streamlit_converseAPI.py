import os
import logging
import boto3
import streamlit as st

from dataclasses import dataclass
from typing import Optional
from botocore.exceptions import ClientError, NoCredentialsError
from config_manager import ConfigManager, ModelInfo
from asyncio_loop_manager import get_or_create_event_loop_thread


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Initialize configuration manager
config_manager = ConfigManager()

@dataclass
class ModelConfig:
    model_id: str
    system_message: str
    messages: list
    max_tokens: int
    budget_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    enable_reasoning: bool = False

def initialize_session_state():
    # Initialize chat history for display
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []

    # Initialize chat history for model interaction
    if "model_messages" not in st.session_state:
        st.session_state.model_messages = []

    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0

    # Initialize image track recorder
    if "file_update" not in st.session_state:
        st.session_state.file_update = False

    if "allow_input" not in st.session_state:
        st.session_state.allow_input = True

    if "enable_reasoning" not in st.session_state:
        st.session_state.enable_reasoning = False

    if "current_model_id" not in st.session_state:
        st.session_state.current_model_id = config_manager.get_default_model()

    # MCP-related session state
    if "mcp_manager" not in st.session_state:
        st.session_state.mcp_manager = None

    if "mcp_initialized" not in st.session_state:
        st.session_state.mcp_initialized = False

    if "tool_config" not in st.session_state:
        st.session_state.tool_config = None

    if "tool_call_history" not in st.session_state:
        st.session_state.tool_call_history = []

    if "mcp_connection_action" not in st.session_state:
        st.session_state.mcp_connection_action = None

    if "mcp_action_result" not in st.session_state:
        st.session_state.mcp_action_result = None

    if "mcp_servers_connected" not in st.session_state:
        st.session_state.mcp_servers_connected = False

    if "previous_mcp_toggle_state" not in st.session_state:
        st.session_state.previous_mcp_toggle_state = False

def reset_app():
    st.session_state.display_messages = []
    st.session_state.model_messages = []
    st.session_state["file_uploader_key"] += 1
    st.session_state.file_update = False
    st.session_state.allow_input = True
    st.rerun()

def check_file_size(file, file_type):
    """
    Check if file size is within limits
    Returns (is_valid, message)
    """
    file_size = len(file.getvalue())
    file_types = config_manager.get_file_types_by_category()

    for category, types in file_types.items():
        if file_type in types:
            file_config = config_manager.get_file_config(category)
            if file_size > file_config.size_limit:
                size_mb = file_config.size_limit / (1024 * 1024)
                return False, f"{category.title()} file '{file.name}' exceeds {size_mb:.1f}MB limit"

    return True, ""

def file_update():
    st.session_state.file_update = True

def allow_input_disable():
    st.session_state.allow_input = False

def stream_multi_modal_prompt(bedrock_runtime, config: ModelConfig, model_info: ModelInfo):
    """
    Original streaming function without tool support.
    Maintained for backward compatibility.
    """
    inference_config = {"maxTokens": config.max_tokens,}
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

    try:
        response = bedrock_runtime.converse_stream(
            modelId=config.model_id,
            messages=config.messages,
            system=[{"text": config.system_message}],
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields
        )

        in_reasoning_block = False
        actual_response_text = ""  # actual response text (excluding reasoning part)
        reasoning_response_text= ""  # reasoning response text
        reasoning_redacted_content = b''
        signature_response_text = ""
        assistant_content = []


        for chunk in response["stream"]:
            if "contentBlockDelta" not in chunk:
                continue

            delta = chunk["contentBlockDelta"]["delta"]

            if "reasoningContent" in delta:
                if "text" in delta["reasoningContent"]:
                    reasoning_text = delta["reasoningContent"]["text"]
                    reasoning_response_text += reasoning_text  # Collect reasoning response text
                    if not in_reasoning_block:
                        yield "----------------\n"
                        in_reasoning_block = True
                    yield reasoning_text
                if "redactedContent" in delta["reasoningContent"]:
                    redacted_content = delta["reasoningContent"]["redactedContent"]
                    reasoning_redacted_content += redacted_content
                if "signature" in delta["reasoningContent"]:
                    signature = delta["reasoningContent"]["signature"]
                    signature_response_text += signature

            elif "text" in delta:
                text = delta["text"]
                actual_response_text += text  # Collect actual response text without reasoning part

                if in_reasoning_block:
                    yield "\n\n----------------\n"
                    in_reasoning_block = False
                yield text

        if reasoning_response_text and signature_response_text:
            assistant_content.append({"reasoningContent": {"reasoningText": {"text": reasoning_response_text, "signature": signature_response_text}}})
        if reasoning_redacted_content:
            assistant_content.append({"reasoningContent": {"redactedContent": reasoning_redacted_content}})
        if actual_response_text:
            assistant_content.append({"text": actual_response_text})

        st.session_state.model_messages.append({"role": "assistant", "content": assistant_content})

    except (ClientError, Exception) as e:
        logger.error(f"ERROR: Can't invoke '{config.model_id}'. Reason: {e}")
        st.error(f"ERROR: Can't invoke '{config.model_id}'. Reason: {e}")
        raise


def get_bedrock_runtime_client(aws_access_key=None, aws_secret_key=None, aws_region=None):
    try:
        if aws_access_key and aws_secret_key and aws_region:
            bedrock_runtime = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
        else:
            bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
    except ClientError as e:
        # Handle errors returned by the AWS service
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"AWS service returned an error: {error_code} - {error_message}")
        st.error(f"AWS service returned an error: {error_code} - {error_message}")
        raise
    except NoCredentialsError:
        # Handle the case where credentials are missing
        logger.error("Unable to retrieve AWS credentials, please check your credentials configuration.")
        st.error("Unable to retrieve AWS credentials, please check your credentials configuration.")
        raise
    except Exception as e:
        # Handle any other unknown exceptions
        logger.error(f"An unknown error occurred: {str(e)}")
        st.error(f"An unknown error occurred: {str(e)}")
        raise
    return bedrock_runtime


async def initialize_mcp_manager():
    """
    Initialize MCP Client Manager and connect to configured servers.
    
    Returns:
        Tuple of (mcp_manager, tool_config, has_error) where has_error is True if initialization failed
    """
    try:
        # Check if mcp.json exists
        if not os.path.exists("mcp.json"):
            logger.info("MCP: mcp.json not found, MCP features disabled")
            return None, None, False

        logger.info("MCP: Initializing MCP Client Manager")

        # Import MCP client
        from mcp_client import MCPClientManager

        # Create manager instance
        mcp_manager = MCPClientManager(config_path="mcp.json")

        # Initialize and connect to servers
        await mcp_manager.initialize()

        # Check if initialization was successful
        if not mcp_manager.config:
            logger.error("MCP: Failed to load MCP configuration")
            return None, None, True

        # Convert tools to Bedrock format
        tool_config = mcp_manager.convert_tools_to_bedrock_format()

        if tool_config:
            tool_count = len(tool_config.get('tools', []))
            logger.info(f"MCP: Initialized with {tool_count} tool(s)")
        else:
            logger.warning("MCP: Initialized but no tools available")

        return mcp_manager, tool_config, False

    except Exception as e:
        # Log detailed error information
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"MCP: Failed to initialize MCP: {type(e).__name__}: {str(e)}")
        logger.error(f"MCP: Full traceback:\n{error_details}")
        return None, None, True


def render_mcp_ui(mcp_manager, model_info=None):
    """
    Render MCP UI with toggle and expander for server details.
    
    Displays toggle to enable/disable MCP servers, and an expander
    showing detailed server status and available tools.
    
    Args:
        mcp_manager: MCPClientManager instance or None
        model_info: ModelInfo instance for the currently selected model
    """
    # Check if MCP is configured
    if mcp_manager:
        # MCP is configured - show functional toggle
        server_status = mcp_manager.get_server_status()
        any_connected = any(
            server["connected"]
            for server in server_status) if server_status else False

        # Update the session state based on actual connection status
        st.session_state.mcp_servers_connected = any_connected

        # Create toggle
        new_mcp_state = st.toggle(
            "MCP Servers",
            value=st.session_state.mcp_servers_connected,
            help="Enable/Disable MCP server connections for tool calling")

        # Detect state change
        if new_mcp_state != st.session_state.mcp_servers_connected:
            if new_mcp_state:
                # User wants to connect
                st.session_state.mcp_connection_action = "reconnect"
            else:
                # User wants to disconnect
                st.session_state.mcp_connection_action = "disconnect"
            st.rerun()

        # Show expander with server details when enabled
        if new_mcp_state:
            render_mcp_servers_expander(mcp_manager, model_info)
    else:
        # MCP is not configured - show disabled toggle with hint
        st.toggle(
            "MCP Servers",
            value=False,
            disabled=True,
            help="MCP not configured. Create mcp.json to enable MCP features.")
        # Show configuration error hint if there was an error
        if hasattr(st.session_state, 'mcp_error') and st.session_state.mcp_error:
            st.caption("⚠️ Configuration error, please check your mcp.json file")


def render_mcp_servers_expander(mcp_manager, model_info=None):
    """
    Render MCP servers with detailed tool information.
    
    Shows a summary header followed by each server as an expander.
    Only displays first 3 tools per server with "... and X more" for the rest.
    
    Args:
        mcp_manager: MCPClientManager instance
        model_info: ModelInfo instance for the currently selected model
    """
    server_status = mcp_manager.get_server_status()

    if not server_status:
        st.caption("ℹ️ No servers configured")
        return

    # Check if current model supports tools
    if model_info and not model_info.supports_tools:
        st.warning("⚠️ Current model does not support tools")
        return

    # Count connected servers and total tools
    connected_count = sum(1 for s in server_status if s['connected'])
    total_tools = sum(len(s.get('tools', [])) for s in server_status if s['connected'])

    # Display summary header in a container with visual grouping
    with st.container():
        # Use caption for a more compact header
        st.caption(f"🔌 **Connected Servers ({connected_count})** - {total_tools} tools available")

        # Display each server in its own expander
        for server in server_status:
            server_name = server['name']
            is_connected = server['connected']
            tools = server.get('tools', [])

            # Create expander for each server
            if is_connected:
                # Connected server - show with green icon
                expander_label = f"🟢 {server_name}: {len(tools)} tools available"
                with st.expander(expander_label, expanded=False):
                    if tools:
                        # Display only first 3 tools
                        tools_to_show = tools[:3]
                        for i, tool in enumerate(tools_to_show):
                            tool_name = tool.get('name', 'Unknown')
                            st.markdown(f"**`{tool_name}`**")

                        # Show "... and X more" if there are more tools
                        if len(tools) > 3:
                            remaining = len(tools) - 3
                            st.caption(f"... and {remaining} more tool(s)")
                    else:
                        st.info("No tools available from this server")
            else:
                # Disconnected server - show with red icon
                error_msg = server.get('error', 'Connection failed')
                expander_label = f"🔴 {server_name} (disconnected)"
                with st.expander(expander_label, expanded=False):
                    st.error(f"❌ {error_msg}")
                    st.caption("Check app.log for details")





def main():
    initialize_session_state()

    # Handle MCP connection actions BEFORE any Streamlit commands
    # This must be done before st.set_page_config()
    if st.session_state.mcp_connection_action and st.session_state.mcp_manager:
        action = st.session_state.mcp_connection_action
        loop_thread = get_or_create_event_loop_thread()

        try:
            if action == "reconnect":
                logger.info("MCP: User requested reconnect all")
                success_count, fail_count = loop_thread.run_coroutine(
                    st.session_state.mcp_manager.reconnect_all()
                )

                # Update tool config
                st.session_state.tool_config = st.session_state.mcp_manager.convert_tools_to_bedrock_format()

                # Store result message for display later
                if success_count > 0 and fail_count > 0:
                    st.session_state.mcp_action_result = {
                        "type": "warning",
                        "message": f"✅ Successfully reconnected {success_count} server(s), ⚠️ Failed: {fail_count}"
                    }
                elif success_count > 0:
                    st.session_state.mcp_action_result = {
                        "type": "success",
                        "message": f"✅ Successfully reconnected {success_count} server(s)"
                    }
                else:
                    st.session_state.mcp_action_result = {
                        "type": "error",
                        "message": f"⚠️ Failed to reconnect {fail_count} server(s)"
                    }

                logger.info(f"MCP: Reconnect complete - Success: {success_count}, Failed: {fail_count}")

            elif action == "disconnect":
                logger.info("MCP: User requested disconnect all")
                disconnected_count = loop_thread.run_coroutine(
                    st.session_state.mcp_manager.disconnect_all()
                )

                # Clear tool config
                st.session_state.tool_config = None

                # Store result message for display later
                if disconnected_count > 0:
                    st.session_state.mcp_action_result = {
                        "type": "info",
                        "message": f"⏸️ Disconnected {disconnected_count} server(s)"
                    }
                else:
                    st.session_state.mcp_action_result = {
                        "type": "info",
                        "message": "ℹ️ No servers were connected"
                    }

                logger.info(f"MCP: Disconnect complete - Disconnected: {disconnected_count}")

        except Exception as e:
            error_msg = f"Error during MCP connection action: {str(e)}"
            logger.error(f"MCP: {error_msg}", exc_info=True)
            st.session_state.mcp_action_result = {
                "type": "error",
                "message": f"❌ {error_msg}"
            }
        finally:
            # Clear the action flag
            st.session_state.mcp_connection_action = None

    # Initialize MCP manager on first run
    if not st.session_state.mcp_initialized:
        try:
            # Get the event loop thread
            loop_thread = get_or_create_event_loop_thread()

            # Run async initialization in the dedicated event loop thread
            mcp_manager, tool_config, has_error = loop_thread.run_coroutine(initialize_mcp_manager())

            st.session_state.mcp_manager = mcp_manager
            st.session_state.tool_config = tool_config
            st.session_state.mcp_initialized = True
            st.session_state.mcp_error = has_error

            if mcp_manager:
                logger.info("MCP: Successfully initialized in main()")
            elif has_error:
                logger.warning("MCP: Initialization completed with errors")
        except Exception as e:
            # Log detailed error information
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"MCP: Unexpected error during MCP initialization: {type(e).__name__}: {str(e)}")
            logger.error(f"MCP: Full traceback:\n{error_details}")
            # Set error flag
            st.session_state.mcp_manager = None
            st.session_state.tool_config = None
            st.session_state.mcp_error = True
            st.session_state.mcp_initialized = True  # Mark as initialized to avoid retry

    # App title
    st.set_page_config(page_title="Bedrock-Streamlit-Chat 💬", page_icon='./utils/logo.png')

    with st.sidebar:
        col1, col2 = st.columns([1,3.5])
        with col1:
            st.image('./utils/logo.png')
        with col2:
            st.title("Bedrock-Streamlit-Chat")

        # Display MCP action result message if available
        if st.session_state.mcp_action_result:
            result = st.session_state.mcp_action_result
            if result["type"] == "success":
                st.success(result["message"])
            elif result["type"] == "warning":
                st.warning(result["message"])
            elif result["type"] == "error":
                st.error(result["message"])
            elif result["type"] == "info":
                st.info(result["message"])
            # Clear the message after displaying
            st.session_state.mcp_action_result = None

        model_names = config_manager.get_all_model_names()
        default_model = config_manager.get_default_model()
        default_index = model_names.index(default_model) if default_model in model_names else 0

        new_model_id = st.selectbox(
            'Choose a Model',
            model_names,
            index=default_index,
            label_visibility="collapsed"
        )

        if new_model_id != st.session_state.current_model_id:
            st.session_state.current_model_id = new_model_id
            reset_app()

        model_info = config_manager.get_model_info(new_model_id)
        model_id = model_info.model_id

        if model_info.supports_reasoning:
            st.session_state.enable_reasoning = st.toggle("Reasoning Mode", value=st.session_state.enable_reasoning,
                                                          help="Enable model's reasoning capability")
        else:
            st.session_state.enable_reasoning = False

        regions = config_manager.get_regions()
        default_region = config_manager.get_default_region()
        default_region_index = regions.index(default_region) if default_region in regions else 0
        aws_region = st.selectbox('Choose a Region', regions, index=default_region_index, label_visibility="collapsed")
        if aws_region:
            os.environ['AWS_REGION'] = aws_region
        else:
            st.error("Please select a valid AWS region")
            return

        with st.expander('AWS Credentials', expanded=False):
            aws_access_key = st.text_input('AWS Access Key', os.environ.get('AWS_ACCESS_KEY_ID', ""), type="password")
            aws_secret_key = st.text_input('AWS Secret Key', os.environ.get('AWS_SECRET_ACCESS_KEY', ""), type="password")

            credentials_changed = (
                aws_access_key != os.environ.get('AWS_ACCESS_KEY_ID', "") or
                aws_secret_key != os.environ.get('AWS_SECRET_ACCESS_KEY', "")
            )

            if st.button('Update AWS Credentials', disabled=not credentials_changed):
                if aws_access_key == "" or aws_secret_key == "":
                    st.warning("Please fill out all the AWS credential fields.")
                else:
                    st.success("AWS credentials are updated successfully!")
                    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
                    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key

        with st.expander('System Prompt', expanded=False):
            system_prompt = st.text_area(
                "System prompt",
                "You are a helpful, harmless, and honest AI assistant. "
                "Your goal is to provide informative and substantive responses to queries while avoiding potential harms.",
                label_visibility="collapsed"
            )

        with st.expander('Model Parameters', expanded=False):
            params_disabled = st.session_state.enable_reasoning

            max_new_tokens = st.number_input(
                min_value=100,
                max_value=model_info.max_tokens,
                step=10,
                value=min(16384, model_info.max_tokens),
                label="Number of tokens to output",
                key="max_new_token"
            )

            if model_info.supports_reasoning and model_info.model_family == "claude":
                budget_tokens = st.number_input(
                    min_value=1024,
                    max_value=65536,
                    step=10,
                    value=4096,
                    label="Number of tokens to think" + (" (disabled in non-reasoning mode)" if not params_disabled else ""),
                    key="budget_tokens",
                    disabled=not params_disabled
                )
            else:
                budget_tokens = None

            col1, col2 = st.columns([4,1])
            with col1:
                temperature = st.slider(
                    min_value=0.1,
                    max_value=1.0,
                    step=0.1,
                    value=0.5,
                    label="Temperature" + (" (disabled in reasoning mode)" if params_disabled else ""),
                    key="temperature",
                    disabled=params_disabled
                )
                top_p_disabled = params_disabled or not model_info.supports_top_p
                top_p_label = "Top P"
                if params_disabled:
                    top_p_label += " (disabled in reasoning mode)"
                elif not model_info.supports_top_p:
                    top_p_label += " (not supported by this model)"
                top_p = st.slider(
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1,
                    value=1.0,
                    label=top_p_label,
                    key="top_p",
                    disabled=top_p_disabled
                )
                if model_info.supports_top_k:
                    top_k = st.slider(
                        min_value=0,
                        max_value=model_info.top_k_max,
                        step=1,
                        value=model_info.top_k_max // 2,
                        label="Top K" + (" (disabled in reasoning mode)" if params_disabled else ""),
                        key="top_k",
                        disabled=params_disabled
                    )
                else:
                    top_k = None

        # MCP Servers toggle and expander (placed after Model Parameters)
        if st.session_state.mcp_initialized:
            render_mcp_ui(st.session_state.mcp_manager, model_info)

        if model_info.supports_multimodal:
            file = st.file_uploader("File Query", accept_multiple_files=True, key=st.session_state["file_uploader_key"], on_change=file_update, help='Support multimodal models', disabled=False)
            file_list = []
            file_types = config_manager.get_file_types_by_category()

            for item in file:
                item_type = item.name.split('.')[-1].lower()

                # Check file size
                is_valid_size, error_message = check_file_size(item, item_type)
                if not is_valid_size:
                    st.error(error_message)
                    return None

                if item_type in file_types["image"]:
                    item_type = 'jpeg' if item_type == 'jpg' else item_type
                    st.image(item, caption=item.name)
                    file_list.append({"image": {"format": item_type, "source": {"bytes": item.getvalue()}}})
                elif item_type in file_types["document"]:
                    file_list.append({"document": {"format": item_type, "name": item.name.split(".")[0], "source": {"bytes": item.getvalue()},  "citations": {"enabled": True}}})
                elif item_type in file_types["video"]:
                    if model_info.supports_video:
                        st.video(item)
                        file_list.append({"video": {"format": item_type, "source": {"bytes": item.getvalue()}}})
                    else:
                        st.error(f"Video files are only supported by models that support video. Please remove {item.name}")
                        return None
                else:
                    st.write(f"Unsupported file type: {item_type}, please remove the file!")
                    return None
        else:
            file = st.file_uploader("File Query", help='Multimodal models only', disabled=True)

        # Clear messages, including uploaded images
        if st.sidebar.button("New Conversation", type="primary"):
            reset_app()

    with st.chat_message("assistant", avatar="./utils/assistant.png"):
        st.write("I am an AI chatbot powered by Amazon Bedrock, what can I do for you？💬")

    # Display chat messages from history on app rerun
    for message in st.session_state.display_messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="./utils/assistant.png"):
                # Handle assistant messages - only display text content
                # Tool use information is already included in the text from streaming
                for item in message["content"]:
                    if "text" in item:
                        st.markdown(item["text"])
        else:
            with st.chat_message(message["role"], avatar="./utils/user.png"):
                for item in message["content"]:
                    if "image" in item:
                        st.image(item["image"]["source"]["bytes"], width=50)
                    elif "document" in item:
                        col1, col2 = st.columns([0.45,8])
                        with col1:
                            st.image('./utils/file.png')
                        with col2:
                            document_full = item["document"]["name"]+"."+item["document"]["format"]
                            st.markdown(document_full)
                    elif "video" in item:
                        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1,1,1,1,1,1,1,1])
                        with col1:
                            st.video(item["video"]["source"]["bytes"])
                    elif "text" in item:
                        st.markdown(item["text"])

    if query := st.chat_input("Input your message...", disabled=not st.session_state.allow_input, on_submit=allow_input_disable):
        # Display user message in chat message container
        with st.chat_message("user", avatar="./utils/user.png"):
            user_content = []
            if st.session_state.file_update:
                file_types = config_manager.get_file_types_by_category()
                for item in file:
                    item_type = item.name.split('.')[-1].lower()
                    if item_type in file_types["image"]:
                        st.image(item, width=50)
                    elif item_type in file_types["video"]:
                        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1,1,1,1,1,1,1,1])
                        with col1:
                            st.video(item)
                    else:
                        col1, col2 = st.columns([0.45,8])
                        with col1:
                            st.image('./utils/file.png')
                        with col2:
                            st.markdown(item.name)
                user_content = file_list
            st.session_state.file_update = False
            st.markdown(query)

        # Add user message to chat history
        user_content.append({"text": query})
        user_message = {"role": "user", "content": user_content}
        st.session_state.display_messages.append(user_message)
        st.session_state.model_messages.append(user_message)

        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="./utils/assistant.png"):
            system_message = system_prompt

            model_messages = st.session_state.model_messages

            model_config = ModelConfig(
                model_id=model_id,
                system_message=system_message,
                messages=model_messages,  # Use model_messages without reasoning part
                max_tokens=max_new_tokens,
                budget_tokens=budget_tokens,
                temperature=temperature if not st.session_state.enable_reasoning else None,
                top_p=top_p if not st.session_state.enable_reasoning and model_info.supports_top_p else None,
                top_k=top_k if not st.session_state.enable_reasoning and model_info.supports_top_k else None,
                enable_reasoning=st.session_state.enable_reasoning
            )

            bedrock_runtime = get_bedrock_runtime_client(
                aws_access_key=os.environ.get('AWS_ACCESS_KEY_ID', ""),
                aws_secret_key=os.environ.get('AWS_SECRET_ACCESS_KEY', ""),
                aws_region=os.environ.get('AWS_REGION', ""))

            with st.spinner('Thinking...'):
                try:
                    # Check if MCP tools are available and model supports tools
                    mcp_manager = st.session_state.mcp_manager
                    tool_config = st.session_state.tool_config

                    if mcp_manager and tool_config and model_info.supports_tools:
                        # Use tool-enabled streaming function (synchronous version)
                        logger.info(f"Using tool-enabled streaming with MCP for model: {model_info.model_id}")

                        # Import synchronous version
                        from stream_tools_sync import stream_conversation_with_tools_sync

                        # Use streamlit's write_stream with sync function
                        response = st.write_stream(
                            stream_conversation_with_tools_sync(
                                bedrock_runtime, model_config, model_info,
                                mcp_manager, tool_config
                            )
                        )
                    else:
                        # Use original streaming function without tools
                        if not model_info.supports_tools and mcp_manager and tool_config:
                            logger.info(f"Model {model_info.model_id} does not support tools, using standard streaming")
                        else:
                            logger.info("Using standard streaming without MCP tools")
                        response = st.write_stream(
                            stream_multi_modal_prompt(
                                bedrock_runtime, model_config, model_info
                            )
                        )

                    if not response:
                        st.error("No response received from the model")
                        st.stop()

                    assistant_content = [{"text": response}]
                    st.session_state.display_messages.append({"role": "assistant", "content": assistant_content})

                except ClientError as err:
                    message = err.response["Error"]["Message"]
                    logger.error("A client error occurred: %s", message)
                    st.error(f"A client error occurred: {message}")
                    st.stop()

                except Exception as e:
                    logger.error(f"An unknown error occurred: {str(e)}")
                    st.error(f"An unknown error occurred: {str(e)}")
                    st.stop()

                finally:
                    st.session_state.allow_input = True
                    st.rerun()

if __name__ == "__main__":
    main()
