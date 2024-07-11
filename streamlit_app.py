import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
import hmac
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader, Settings

st.set_page_config(page_title="Chat with your AI Virtual Assistant, powered by LlamaIndex",
                   page_icon="🔥",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

st.title("AI Virtual Assistant")

def check_password():

    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets.mypassword):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False
    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()

openai.api_key = st.secrets.openai_key


if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Start your literature review",
        }
    ]

st.text('Preparing the model...')
Settings.llm = OpenAI(
            model="gpt-4o",
            temperature=0.2,
            system_prompt="""You are my AI Virtual 
            Assistant to write literature review.
            Assume that all questions are related 
            to the science and economics. Keep 
            your answers technical, academic 
            languages and based on 
            facts – do not hallucinate features.
            Template of the literature review:
            Introductiom
            State of the art
            add here the ad-hoc paragraph
            References

            Important: use an academic languages
            """,
)

@st.cache_resource(show_spinner=False)
def load_data():
    with st.expander('See process'):
        st.text("Load custom docs...")
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True, num_workers=4)
        docs = reader.load_data()
        number_of_documents = len(docs)
        st.text(f"{number_of_documents} documents loaded")
        st.text("Prepare the index...")
        index = PropertyGraphIndex.from_documents(docs)
        st.text("Index is ready")
    return index

st.text('Loading your data...')
index = load_data()
st.text('Preparing the engine...')

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

st.text('Ready...')

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
