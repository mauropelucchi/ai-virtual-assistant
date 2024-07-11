import streamlit as st
import openai
import requests
import json
from llama_index.llms.openai import OpenAI
import hmac
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex, \
                             SimpleDirectoryReader, \
                             load_index_from_storage, \
                             Settings, \
                             StorageContext, \
                             Document
import os

def get_academic_papers_from_dblp(query: str):
    query = query.replace(" ", "+")
    feeds_summary = []
    url = f'https://dblp.org/search/publ/api?q={query}&format=json'
    response = requests.get(url)
    os.write(1,response.text)
    data = response.json()
    feeds = data["result"]["hits"]["hit"]
    for feed in feeds:
        feeds_summary.append(
            Document(
                text=feed['title'],
                metadata={"author": feed["info"], "score": feed['@score']},
            )
    )
    return feeds_summary

persist_directory = './index'
index_files = ['vector_store.json', 'docstore.json', 'index_store.json']
index_exists = all(os.path.exists(os.path.join(persist_directory, file)) for file in index_files)

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
st.subheader("Welcome to the AI Virtual Assistant for Literature Review")
st.text("The assistant is based on https://dblp.org/")
st.text("Example of prompts:")
st.text("prepare literature review on labour market and artificial intelligence")

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
            model="gpt-4",
            temperature=0.2,
            system_prompt="""You are my AI Virtual 
            Assistant to write literature review.
            Assume that all questions are related 
            to the science and economics. Keep 
            your answers technical, academic 
            languages and based on 
            facts – do not hallucinate features.
            Template of the literature review:
            Important: add an Introduction session
            Important: References
            Important: use an academic languages
            """,
)

@st.cache_resource(show_spinner=False)
def load_data():
    with st.expander('See process'):
        if not index_exists:
            st.text("Loading new documents...")
            docs = SimpleDirectoryReader(input_dir="./data").load_data()
            number_of_documents = len(docs)
            st.text(f"{number_of_documents} documents loaded")
            st.text("Preparing the index...")
            index = VectorStoreIndex.from_documents(docs, show_progress=True)
            index.storage_context.persist(persist_dir="persist_directory")
        else:
            st.text("Loading the index...")
            storage_context = StorageContext.from_defaults(persist_dir=persist_directory)
            index = load_index_from_storage(storage_context)
            docs = SimpleDirectoryReader(input_dir="./data").load_data()

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
        with st.expander('See process'):
            response = st.session_state.chat_engine.stream_chat("""
                Generate a list of relevant terms (max 10) to
                retrieve relevant documents
                format of the output:
                term 1####
                term 2####
                term 3####
            """)
            response_text = ""
            for token in response.response_gen:
                response_text = response_text + " " + token
            parser = SimpleNodeParser()
            for response in response_text.split('####'):
                st.text(f"Downloading new relevant documents about {response}...")
                new_documents = get_academic_papers_from_dblp(response)
                st.text("Adding new docs to the existing index...")
                index.insert_nodes(parser.get_nodes_from_documents(new_documents))
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        st.session_state.messages.append(message)
