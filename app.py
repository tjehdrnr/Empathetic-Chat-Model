import base64
import time
import logging

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

user_avatar = "images/user2.png"
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
        logger.info(f"Added user: {user_name}")
        st.rerun()

    if not user_name and submitted:
        st.error("The user name is empty. Please check your user name.")


@st.cache_data(show_spinner=False)
def welcome_message():
    welcome_msg = "안녕하세요! 오늘은 어떤 하루를 보내고 계신가요?🥰"
    with st.chat_message('assistant', avatar=assistant_avatar):
        st.markdown(welcome_msg)


def previous_messages(controller):
    """Print all previous messages"""
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

    with st.expander("STORED MESSAGES INFO", icon=':material/info:', expanded=True):
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



def main():
    """
    Display streamlit updates and handle the chat interface.
    """
    st.session_state.setdefault('user_name', None)
    st.session_state.setdefault('delete_id', None)
    
    if st.session_state.user_name is not None:
        controller = initialize_session_state(st.session_state.user_name)

    if st.session_state.user_name is None:
        login_page()
    else:
        # Inferface of sidebar.
        st.title("Streamlit Empathetic Chatbot")
        with st.sidebar:
            sidebar_logo()
            st.sidebar.markdown('---')
            messages_info(controller)
            
            inference_kwargs = inference_settings()
        
        # User interactive menu widgets.
        with st.container():
            clear, save = st.columns([1, 1])
            with clear:
                if st.button(":material/clear_all:", help="Clear all messages and histories"):
                    controller.clear_all()
            # with save:
            #     contents = controller.save_messages()
            #     st.download_button(":material/save:", contents, help="Download message contents as .txt file")

        # Starting point of the main chat interface.
        welcome_message()
        previous_messages(controller)
        
        if user_input := st.chat_input("어떤 고민이든 말씀해보세요."):
            message = controller.docstore.add('user', user_input)

            # User's chat inferface.
            with st.chat_message('user', avatar=user_avatar):
                col1, col2 = st.columns([9, 1])
                with col1:
                    st.markdown(user_input)
                with col2:
                    key = message.metadatas['_id']
                    if st.button(':material/delete:', key=key, help="Delete this conversation"):
                        st.session_state.delete_id = key

            response = controller.get_response(user_input, **inference_kwargs)
            controller.docstore.add('assistant', response)

            # Assistant's chat interface.
            with st.container():
                with st.chat_message('assistant', avatar=assistant_avatar):
                    st.write_stream(streamer(response))

            controller.retriever.add_to_index(controller.docstore.history[-1])
            st.rerun()
        
        # If users press the delete button, delete messages and history immediately.
        # This process includes removing indexed vector's id also.
        if st.session_state.delete_id is not None: 
            controller.delete_chat(st.session_state.delete_id)
            st.session_state.delete_id = None
            st.rerun()

        

        # message 용량 알고리즘 짜야함..
        # json, csv, tsv 등 파일 셀렉트하고, 저장하는 기능 추가
        

        
        # 답변 두개(DPO 모드) 받아서 preference 체크하고 저장하기.. 인메모리니까 파일 불러와서 이어서 write 하는 방식으로    
        

        
        #깃허브 링크 달기??

        




if __name__ == "__main__":
    main()





