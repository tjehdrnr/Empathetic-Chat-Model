import csv
import base64
import time, os
import logging
import pandas as pd

import streamlit as st
from backend.controller import AppController

logging.basicConfig(encoding="utf-8", level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configurations
st.set_page_config(
    page_title="Streamlit Empathetic Chatbot",
    page_icon="assistant.png",
    layout="centered",
    initial_sidebar_state="auto",
)

user_avatar = "images/user.png"
assistant_avatar = "images/assistant.png"
image_path = "images/logo.png"


@st.cache_resource
def initialize_session_state(user_id: str):
    return AppController(user_id)


def streamer(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


@st.dialog("Login Page")
def login_page():
    user_name = st.text_input("User name")
    submitted = st.button("Submit")
    st.warning("Please enter your user name.")

    if user_name and submitted:
        st.session_state.user_name = user_name
        logger.info(f"Logged-in user: {user_name}")
        st.rerun()

    if not user_name and submitted:
        st.error("The user name is empty. Please check your user name.")


@st.cache_data(show_spinner=False)
def welcome_message():
    welcome_msg = "안녕하세요! 오늘은 어떤 하루를 보내고 계신가요?🥰"
    with st.chat_message('assistant', avatar=assistant_avatar):
        st.markdown(welcome_msg)


def previous_messages(controller):
    """Display all previous chat messages"""
    for obj in controller.docstore.messages:
        with st.chat_message(
            obj.message['role'],
            avatar=user_avatar if obj.message['role'] == 'user' else assistant_avatar
        ):
            if obj.message['role'] == 'user':
                col1, col2 = st.columns([9, 1])
                with col1:
                    st.markdown(obj.message['content'])
                with col2:
                    key = obj.metadatas['_id']
                    if st.button(':material/delete:', key=key, help="Delete this conversation."):
                        st.session_state.delete_id = key
            else:
                st.markdown(obj.message['content'])


@st.cache_data(show_spinner=False)
def image_to_base64(image_path):
    """Convert image to base64"""
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        return None


def sidebar_logo():
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
    image_base64 = image_to_base64(image_path)
    if image_base64:
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{image_base64}" class="cover-glow">',
            unsafe_allow_html=True,
        )


def messages_info(controller):
    info = controller.docstore.count()

    with st.expander("STORED MESSAGES INFO", icon=':material/info:', expanded=False):
        st.markdown(f"Number of messages: {info['n_messages']}")
        st.markdown(f"Number of histories: {info['n_history']}")
        st.write(":material/person: ***USER INPUT***")
        st.markdown(f"Average token length: {info['user_avg']}")
        st.markdown(f"Maximum token length: {info['user_max']}")
        st.markdown(f"Minimum token length: {info['user_min']}")
        st.write(":material/robot_2: ***RESPONSE***")
        st.markdown(f"Average token length: {info['assistant_avg']}")
        st.markdown(f"Maximum token length: {info['assistant_max']}")
        st.markdown(f"Minimum token length: {info['assistant_min']}")


def inference_settings():
    inference_args = {}

    with st.expander("INFERENCE SETTINGS", icon=':material/settings:', expanded=True):
        k = st.select_slider(
            "k", options=range(1, 6), value=2,
            help="Number of histories to use for previous conversation.")
        inference_args['k'] = k

        time_weight = st.slider(
            "time weight", min_value=0.0, max_value=1.0, value=0.2, step=0.05,
            help="Adjust time weight to similar vector search. The closer it is to 1, the more impact it has.")
        inference_args['time_weight'] = time_weight

        repetition_penalty = st.slider(
            "repetition penalty", min_value=1.0, max_value=10.0, value=1.5, step=0.1,
            help=r"""Penalize the model for generating the same token multiple times.
                The higher the value, the stronger the penalty applied to repeated tokens.
                The value should be greater than 1.0 to have any effect.
                """)
        inference_args['repetition_penalty'] = repetition_penalty

        min_new_tokens, max_new_tokens = st.select_slider(
            "Min & Max new tokens",
            options=[2 ** i for i in range(12)],
            value=(16, 256),
            help="Range of token lengths to be generated by the model.")
        inference_args['min_new_tokens'] = min_new_tokens
        inference_args['max_new_tokens'] = max_new_tokens

        st.markdown('---')
        early_stopping = st.toggle(
            "Use early stopping",
            value=True,
            help="The text generation will stop as soon as the model generates an end-of-sequence token (eos_token_id).")
        inference_args['early_stopping'] = early_stopping

        do_sample = st.toggle(
            "Do sampling",
            value=True,
            help="Generate text by randomly selecting the next token from the probability distribution. When activate sampling, several other parameters can be use:\n" +
                "1. temperature: A higher temperature (e.g., 1.0 or higher) results in more random outputs, while a lower temperature (closer to 0.0) makes the model's output more focused and deterministic.\n" +
                "2. top_k: Limits the sampling pool to the top k tokens with the highest probabilities. For instance, if top_k=50, the model will only consider the 50 most likely tokens for the next word.\n" +
                "3. top_p: Limits the sampling pool to the smallest set of tokens whose cumulative probability is greater than or equal to p. For example, if top_p=0.9, the model will consider only the tokens that together have a cumulative probability of 90%.")
        inference_args['do_sample'] = do_sample

        if do_sample:
            temperature = st.slider("temperature", min_value=0.0, max_value=1.0, value=0.25)
            top_k = st.slider("top_k", min_value=0, max_value=100, value=30)
            top_p = st.slider("top_p", min_value=0.0, max_value=1.0, value=0.9)

            inference_args['temperature'] = temperature
            inference_args['top_k'] = top_k
            inference_args['top_p'] = top_p

    return inference_args


def save_dpo_data(controller):
    """
    Save the stored DPO data in the document store as a .tsv file at the specified directory.
    """
    
    save_dir = controller.config.save_dir
    save_fn = os.path.join(save_dir, "dpo_data.tsv")

    controller.docstore.dpo_data.replace("\n+", " ", regex=True, inplace=True)

    if os.path.exists(save_fn):
        original_data = pd.read_csv(save_fn, sep='\t', encoding='utf-8')
        merged_data = pd.concat([original_data, controller.docstore.dpo_data])
        merged_data = merged_data.reset_index(drop=True)
        merged_data.to_csv(
            save_fn, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE, index=False
        )
    else:
        controller.docstore.dpo_data.to_csv(
            save_fn, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE, index=False
        )



def main():
    """
    Display streamlit updates and handle the chat interface
    """
    st.session_state.setdefault('user_name', None)
    st.session_state.setdefault('delete_id', None)
    st.session_state.setdefault('responses', [])
    st.session_state.setdefault('context', None)
    st.session_state.setdefault('chosen', None)

    if st.session_state.user_name is not None:
        controller = initialize_session_state(st.session_state.user_name)
        
    if st.session_state.user_name is None:
        login_page()
    else:
        # Inferface of sidebar
        st.title("Streamlit Empathetic Chatbot")
        with st.sidebar:
            sidebar_logo()
            st.sidebar.markdown('---')
            messages_info(controller)
            inference_kwargs = inference_settings()
        
        # User interactive widgets
        with st.container():
            clear, save, dpo, _ = st.columns([1, 1, 3, 5])
            with clear:
                if st.button(":material/refresh:", key="clear", help="Start a new chat"):
                    controller.clear_all()
                    st.session_state.responses = []
                    st.session_state.context = None
                    st.session_state.chosen = None
            with save:
                if st.button(":material/save:", key="save", help="Save DPO data"):
                    save_dpo_data(controller)
            with dpo: 
                dpo_mode = st.toggle("DPO Mode")
                if dpo_mode:
                    inference_kwargs['dpo_mode'] = True

        # Starting point of the main chat interface
        welcome_message()
        previous_messages(controller)

        if user_input := st.chat_input("어떤 고민이든 말씀해보세요."):
            message = controller.docstore.add('user', user_input)
            # User input inferface
            with st.chat_message('user', avatar=user_avatar):
                col1, col2 = st.columns([9, 1])
                with col1:
                    st.markdown(user_input)
                with col2:
                    key = message.metadatas['_id']
                    if st.button(':material/delete:', key=key, help="Delete this conversation"):
                        st.session_state.delete_id = key
            
            response, history = controller.get_response(user_input, **inference_kwargs)

            # When DPO mode is activated, two responses are displayed in the chat interface.
            # Each response is sampled during sampling process.
            if dpo_mode and isinstance(response, tuple):
                with st.container():
                    res1, res2 = st.columns([5, 5])
                    with res1:
                        with st.container(border=True):
                            st.write_stream(streamer(response[0]))
                            st.session_state.responses.append(response[0])
                    with res2:
                        with st.container(border=True):
                            st.write_stream(streamer(response[1]))
                            st.session_state.responses.append(response[1])

                st.session_state.context = controller.get_context(history, user_input)
            else:
                # Assistant response interface
                with st.container():
                    with st.chat_message('assistant', avatar=assistant_avatar):
                        st.write_stream(streamer(response))
                    controller.docstore.add('assistant', response)
                    controller.retriever.add_to_index(controller.docstore.history[-1])

        # The User select which response is better.
        # The selected response is stored as 'chosen' for the next generation,
        # and the other response is stored as 'rejected' in the DPO dataset.
        if dpo_mode and st.session_state.responses:
            with st.container():
                col1, col2 = st.columns([5, 5])
                with col1:
                    with st.container():
                        btn1 = st.button(
                            ":material/check:", key="res1", use_container_width=True,
                            help="Select Left Response")
                        if btn1:
                            st.session_state.chosen = 0
                with col2:
                    with st.container():
                        btn2 = st.button(
                            ":material/check:", key="res2", use_container_width=True,
                            help="Select Right Response")
                        if btn2:
                            st.session_state.chosen = 1

                if btn1 or btn2:
                    chosen_idx = st.session_state.chosen
                    controller.docstore.add('assistant', st.session_state.responses[chosen_idx])
                    controller.retriever.add_to_index(controller.docstore.history[-1])
                    controller.write_dpo_data(
                        context=st.session_state.context,
                        chosen=st.session_state.responses[chosen_idx],
                        rejected=st.session_state.responses[1 - chosen_idx],
                    )
                    st.session_state.responses = []
                    st.toast("Selection Completed! :material/emoticon:")
                    st.rerun()

        # If user press the delete button, delete messages and history immediately.
        # This process includes removing indexed vector's id also.
        if st.session_state.delete_id is not None: 
            controller.delete_chat(st.session_state.delete_id)
            st.session_state.delete_id = None
            st.rerun()



if __name__ == "__main__":
    main()