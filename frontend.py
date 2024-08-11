import os, json, time
import logging
from datetime import datetime
import base64
import uuid

from dotenv import load_dotenv
from argparse import ArgumentParser
from arguments import Arguments

import streamlit as st
from streamlit_float import *
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings


load_dotenv()

logger = logging.getLogger()
logging.basicConfig(encoding="utf-8", level=logging.INFO)

stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

PREFIX_USER = "### 명령어: "
PREFIX_ASSISTANT = "### 응답: "

# Streamlit page configurations
st.set_page_config(
    page_title="Streamlit Emphathetic Chatbot",
    page_icon="logo.png",
    initial_sidebar_state="auto",
)
st.title("Streamlit Emphathetic Chatbot")

def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        return None
    

@st.cache_data()
def load_prompt_template(_config):
    template_fn = os.path.join(os.getcwd(), _config.template_path)

    with open(template_fn, "r", encoding="utf-8") as f:
        prompt_template = json.load(f).get("prompt")
    
    return prompt_template


def get_response(user_input, model, config):
    prompt_template =load_prompt_template(config)

    history = '\n'.join(
        list(map(
            lambda x: PREFIX_USER + x["content"] 
            if x["role"] == "user" else PREFIX_ASSISTANT + x["content"], 
            st.session_state.messages
            ))
    )
    prompt = prompt_template.format(
        history=history,
        instruction=user_input
    )
    response = model.invoke(prompt).content
    return response


def streaming(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.1)


@st.dialog("🎨 Upload a picture")
def upload_document():
    st.warning(
        "This is a demo dialog window. You need to process the file afterwards.",
        icon="💡",
    )
    picture = st.file_uploader(
        "Choose a file", type=["jpg", "png", "bmp"], label_visibility="hidden"
    )
    if picture:
        st.session_state["uploaded_pic"] = True
        st.rerun()


# This function logs the last question and answer in the chat messages
def log_feedback(icon):
    st.toast("소중한 의견 감사합니다!", icon="👌")

    # We retrieve the last question and answer
    last_messages = json.dumps(st.session_state.messages[-2:])

    # We record the timestamp
    activity = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": "

    # And include the messages
    activity += "positive" if icon == "👍" else "negative"
    activity += ": " + last_messages

    # And log everything
    logger.info(activity)


def add_message(role, content):
    message_id = str(uuid.uuid4())
    st.session_state.messages.append(
        {"id": message_id, "role": role, "content": content},
    )
    return message_id

def log_feedback(messaged_id, feedback):
    st.session_state.feedback[messaged_id] = feedback
    st.toast


def main(model, config: ArgumentParser):
    """
    Display streamlit updates and handle the chat interface.
    """
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "uploaded_pic" in st.session_state and st.session_state.uploaded_pic:
        st.toast("Picture uploaded!", icon="📥")
        del st.session_state.uploaded_pic

    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    
    user_avatar = "👩‍💻"
    assistant_avatar = "🤖"

    # In case of rerun of the last question, we remove the last answer
    # from st.session_state.messages
    if "rerun" in st.session_state and st.session_state.rerun:
        st.session_state.messages.pop(-1)

    welcome_message = "안녕하세요! 오늘은 어떤 하루를 보내고 계신가요?🥰"
    with st.chat_message("assistant", avatar=assistant_avatar):
        st.markdown(welcome_message)
    
    # Insert custom CSS for glowing effect
    st.markdown(
        """
        <style>.cover-glow {width: 60%; height: auto; padding: 3px;
            box-shadow: 0 0 5px #331a00, 0 0 10px #663300, 0 0 15px #994d00,
                        0 0 20px #cc6600, 0 0 25px #ff8000, 0 0 30px #ff9933, 0 0 35px #ffb366;
            position: relative; z-index: -1; border-radius: 45px; display: block;
            margin-left: auto; margin-right: auto;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load and display sidebar image
    img_path = "logo.png"
    img_base64 = img_to_base64(img_path)
    if img_base64:
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
            unsafe_allow_html=True,
        )
    st.sidebar.markdown("---")

    for message in st.session_state.messages:
        with st.chat_message(
            message["role"],
            avatar=assistant_avatar if message["role"] == "assistant" else user_avatar,
            ):
            st.markdown(message["content"])

    if user_input := st.chat_input("어떤 고민이든 말씀해보세요."):
        with st.chat_message("user"):
            st.markdown(user_input)
    
    if user_input or ("rerun" in st.session_state and st.session_state.rerun):
        with st.chat_message("assistant"):
            response = get_response(user_input, model, config)
            st.write_stream(streaming(response))

        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        if "rerun" in st.session_state and st.session_state.rerun:
            st.session_state.rerun = False

    # The button will trigger the logging function
    col1, col2, col3 = st.columns((12, 1, 1))
    with col1:
        st.markdown(
            """
            <p style='font-size:12px; color:rgba(0, 0, 0, 0.5); text-align:right;'>
                위 응답은 어떠셨나요? 피드백을 남겨주세요.
            </p>
            """,
            unsafe_allow_html=True
        )
    with col2:
        if st.button("👍", help="좋아요", use_container_width=True):
            st.session_state.feedback_clicked = "👍"
    with col3:
        if st.button("👎", help="별로예요", use_container_width=True):
            st.session_state.feedback_clicked = "👎"
    # Custom CSS for the buttons
    st.markdown(
        """
        <style>
        button[title='좋아요'], button[title='별로예요'] {
            font-size: 5px;
            padding: 4px 4px;
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if len(st.session_state["messages"]) > 0:
        action_buttons_container = st.container()
        action_buttons_container.float(
            "bottom: 6.9rem;background-color: var(--default-backgroundColor); padding-top: 1rem;"
        )

        # We set the space between the icons thanks to a share of 100
        # cols_dimensions = [7, 14.9, 14.5, 9.1]
        # cols_dimensions.append(100 - sum(cols_dimensions))

        # col1, col2, col3, col4 = action_buttons_container.columns(
        #     cols_dimensions
        # )

        # with col1:
        #     # Converts the list of messages into a JSON format
        #     json_messages = json.dumps(st.session_state.messages).encode("utf-8")
        #     # And the corresponding download button
        #     st.download_button(
        #         label="📥 Save",
        #         data=json_messages,
        #         file_name="chat_conversation.json",
        #         mime="application/json",
        #     )
        
        # with col2:
        #     # We set the message back to 0 and rerun the app
        #     # (this part could probably be improved with the cache option)
        #     if st.button("🧹 Clear"):
        #         st.session_state["messages"] = []

        #         if "uploaded_pic" in st.session_state:
        #             del st.session_state["uploaded_pic"]

        #         st.rerun()
        
        # with col3:
        #     if st.button("🎨 Upload"):
        #         upload_document()
        
        # with col4:
        #     icon = "🔁 Rerun"
        #     if st.button(icon):
        #         st.session_state["rerun"] = True
        #         st.rerun()
    else:
        # At the first run of a session, we temporarly display a message
        if "disclaimer" not in st.session_state:
            with st.empty():
                for seconds in range(3):
                    st.warning(
                        "버튼(👍 또는 👎)을 클릭하여 챗봇 응답에 대한 피드백을 줄 수 있습니다. 이를 통해, 손쉽게 사용자의 선호 응답 데이터셋을 구축할 수 있습니다.",
                        icon="💡",
                    )
                    time.sleep(3)
                st.write("")
                st.session_state["disclaimer"] = True

            

        

if __name__ == "__main__":
    config = Arguments.demo_args()
    model = ChatOpenAI()
    main(model, config)
                
    