import os
import json
import base64
import logging
import boto3
import hmac
import streamlit as st

from botocore.exceptions import ClientError, NoCredentialsError


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# The default credential is admin/123qaz, you can set those with your own at .streamlit/secrets.toml

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False

if not check_password():
    st.stop()

def file_update():
    st.session_state.file_update = True

def allow_input_disable():
    st.session_state.allow_input = False

def stream_multi_modal_prompt(bedrock_runtime, model_id, system_message, messages, max_tokens, temperature, top_p, top_k):
    inference_config = {
        "maxTokens": max_tokens,
        "temperature": temperature,
        "topP": top_p
    }
    additional_model_fields = {"top_k": top_k}

    try:
        response = bedrock_runtime.converse_stream(
            modelId=model_id,
            messages=messages,
            system=[{"text": system_message}],
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields
        )

        for chunk in response["stream"]:
            if "contentBlockDelta" in chunk:
                text = chunk["contentBlockDelta"]["delta"]["text"]
                yield text
    except (ClientError, Exception) as e:
        logger.error(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
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
            bedrock_runtime = boto3.client('bedrock-runtime')
    except ClientError as e:
        # Handle errors returned by the AWS service
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"AWS service returned an error: {error_code} - {error_message}")
        raise
    except NoCredentialsError:
        # Handle the case where credentials are missing
        logger.error("Unable to retrieve AWS credentials, please check your credentials configuration.")
        raise
    except Exception as e:
        # Handle any other unknown exceptions
        logger.error(f"An unknown error occurred: {str(e)}")
        raise
    return bedrock_runtime

def main():
    # App title
    st.set_page_config(page_title="Bedrock-Claude-Chat 💬", page_icon='./utils/logo.png')

    with st.sidebar:
        col1, col2 = st.columns([1,3.5])
        with col1:
            st.image('./utils/logo.png')
        with col2:
            st.title("Bedrock-Claude-Chat")
        
        with st.expander('AWS Credentials', expanded=False):
            aws_access_key = st.text_input('AWS Access Key', os.environ.get('AWS_ACCESS_KEY_ID', ""), type="password")
            aws_secret_key = st.text_input('AWS Secret Key', os.environ.get('AWS_SECRET_ACCESS_KEY', ""), type="password")
            aws_region = st.text_input('AWS Region', os.environ.get('AWS_REGION', ""))

            credentials_changed = (
                aws_access_key != os.environ.get('AWS_ACCESS_KEY_ID', "") or
                aws_secret_key != os.environ.get('AWS_SECRET_ACCESS_KEY', "") or
                aws_region != os.environ.get('AWS_REGION', "")
            )

            if st.button('Update AWS Credentials', disabled=not credentials_changed):
                if aws_access_key == "" or aws_secret_key == "" or aws_region == "":
                    st.warning("Please fill in all the AWS credential fields.")
                else:
                    st.success("AWS credentials are updated successfully!")
                    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
                    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
                    os.environ['AWS_REGION'] = aws_region

        model_id = st.selectbox('Choose a Model', ('Anthropic Claude-V3-Haiku', 'Anthropic Claude-V3-Sonnet', 'Anthropic Claude-V2.1', 'Anthropic Claude-V2', 'Anthropic Claude-Instant-V1.2'), index=1, label_visibility="collapsed")
        model_id = {
            'Anthropic Claude-V2': 'anthropic.claude-v2',
            'Anthropic Claude-V2.1': 'anthropic.claude-v2:1',
            'Anthropic Claude-Instant-V1.2': 'anthropic.claude-instant-v1',
            'Anthropic Claude-V3-Haiku': 'anthropic.claude-3-haiku-20240307-v1:0',
            'Anthropic Claude-V3-Sonnet': 'anthropic.claude-3-sonnet-20240229-v1:0',
        }.get(model_id, model_id)

        with st.expander('System Prompt', expanded=False):
            system_prompt = st.text_area(
                "System prompt", 
                "You are a helpful, harmless, and honest AI assistant. "
                "Your goal is to provide informative and substantive responses to queries while avoiding potential harms.", 
                label_visibility="collapsed"
            )

        with st.expander('Model Parameters', expanded=False):
            max_new_tokens= st.number_input(
                min_value=10,
                max_value=4096,
                step=10,
                value=2048,
                label="Number of tokens to generate",
                key="max_new_token"
            )
            col1, col2 = st.columns([4,1])
            with col1:
                temperature = st.slider(
                    min_value=0.1,
                    max_value=1.0,
                    step=0.1,
                    value=0.5,
                    label="Temperature",
                    key="temperature"
                )
                top_p = st.slider(
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1,
                    value=1.0,
                    label="Top P",
                    key="top_p"
                )
                top_k = st.slider(
                    min_value=0,
                    max_value=500,
                    step=1,
                    value=250,
                    label="Top K",
                    key="top_k"
                )

        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 0
            
        if "claude-3" in model_id:
            file = st.file_uploader("File Query", accept_multiple_files=True, key=st.session_state["file_uploader_key"], on_change=file_update, help='Claude-V3 only', disabled=False)
            file_list = []

            image_types = {'gif', 'jpg', 'jpeg', 'png', 'webp'}
            document_types = {'pdf', 'csv', 'doc', 'docx', 'xls', 'xlsx', 'html', 'txt', 'md'}

            for item in file:
                item_type = item.name.split('.')[-1]
                if item_type in image_types:
                    item_type = 'jpeg' if item_type == 'jpg' else item_type
                    st.image(item, caption=item.name)
                    file_list.append({"image": {"format": item_type, "source": {"bytes": item.getvalue()}}})
                elif item_type in document_types:
                    file_list.append({"document": {"format": item_type, "name": item.name.split(".")[0], "source": {"bytes": item.getvalue()}}})
                else:
                    st.write(f"Unsupported file type: {item_type}, please remove the file!")
                    return None
        else:
            file = st.file_uploader("File Query", help='Claude-V3 only', disabled=True)
    
        # Clear messages, including uploaded images
        if st.sidebar.button("New Conversation", type="primary"):
            st.session_state.messages = []
            st.session_state.allow_input = True
            st.empty()
            st.session_state["file_uploader_key"] += 1
            st.rerun()

    with st.chat_message("assistant", avatar="./utils/assistant.png"):
        st.write("I am an AI chatbot powered by Amazon Bedrock Claude, what can I do for you？💬")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize image track recorder
    if "file_update" not in st.session_state:
        st.session_state.file_update = False

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="./utils/assistant.png"):
                st.markdown(message["content"][0]["text"])
        else:
            with st.chat_message(message["role"], avatar="./utils/user.png"):
                for item in message["content"]:
                    if "image" in item:
                        image_data = item["image"]["source"]["bytes"]
                        st.image(image_data, width=50)
                    elif "document" in item:
                        col1, col2 = st.columns([0.45,8])
                        with col1:
                            st.image('./utils/file.png')
                        with col2:
                            document_full = item["document"]["name"]+"."+item["document"]["format"]
                            st.markdown(document_full)
                    else:
                        st.markdown(item["text"])

    if "allow_input" not in st.session_state:
        st.session_state.allow_input = True

    if query := st.chat_input("Input your message...", disabled=not st.session_state.allow_input, on_submit=allow_input_disable):
        # Display user message in chat message container
        with st.chat_message("user", avatar="./utils/user.png"):
            user_content = []
            if st.session_state.file_update:
                for item in file:
                    item_type = item.name.split('.')[-1]
                    if item_type in image_types:
                        st.image(item, width=50)
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
        st.session_state.messages.append({"role": "user", "content": user_content})
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="./utils/assistant.png"):
            system_message = system_prompt
            messages = st.session_state.messages
            bedrock_runtime = get_bedrock_runtime_client(
                aws_access_key=os.environ.get('AWS_ACCESS_KEY_ID', ""), 
                aws_secret_key=os.environ.get('AWS_SECRET_ACCESS_KEY', ""), 
                aws_region=os.environ.get('AWS_REGION', ""))
            with st.spinner('Thinking...'):
                try:
                    response= st.write_stream(stream_multi_modal_prompt(
                        bedrock_runtime, model_id, system_message, messages, max_new_tokens, temperature, top_p, top_k
                        )
                    )
                    assistant_content = [{"text": response}]
                    st.session_state.messages.append({"role": "assistant", "content": assistant_content})
                except ClientError as err:
                    message = err.response["Error"]["Message"]
                    logger.error("A client error occurred: %s", message)
                    st.error(f"A client error occurred: {message}")
                except Exception as e:
                    logger.error(f"An unknown error occurred: {str(e)}")
                    st.error(f"An unknown error occurred: {str(e)}")
                finally:
                    st.session_state.allow_input = True
                    st.rerun()
            
if __name__ == "__main__":
    main()