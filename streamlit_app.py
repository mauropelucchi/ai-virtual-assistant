import streamlit as st
import fitz
import openai
import requests
from llama_index.llms.openai import OpenAI
import hmac
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex, \
                             SimpleDirectoryReader, \
                             load_index_from_storage, \
                             Settings, \
                             StorageContext, \
                             Document
import arxiv

client = arxiv.Client()

html_temp = """
            <div style="background-color:{};padding:1px">
            
            </div>
            """

def get_academic_papers_from_dblp(query: str):
    query = query.replace(" ", "+")
    feeds_summary = []
    try:
        url = f'https://dblp.org/search/publ/api?q={query}&format=json'
        response = requests.get(url)
        data = response.json()
        hits = data["result"]["hits"]["hit"]
        for hit in hits:
            authors = ""
            author_info = hit["info"]["authors"]["author"]
            if isinstance(author_info, list):
                for author in author_info:
                    authors = authors + "," + author["text"]
            else:
                authors = author_info["text"]
                    
            feeds_summary.append(
                Document(
                    text=hit['info']['title'],
                    metadata={"author": authors, "score": hit['@score']},
                )
        )
    except:
        pass
    return feeds_summary

def get_arxiv_documents(query):
    feeds_summary = []
    search = arxiv.Search(
        query = query,
        max_results = 10,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )
    results = client.results(search)
    for article in results:
        authors = ""
        for author in article.authors:
            authors = author.name + ", " + authors
        feeds_summary.append(
                Document(
                    text=article.summary,
                    metadata={"author": authors, "title": article.title},
                )
    )
    return feeds_summary

st.set_page_config(page_title="Chat with your AI Virtual Assistant, powered by LlamaIndex",
                   page_icon="🔥",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

st.title("AI Virtual Assistant")
parser = SimpleNodeParser()
index = None

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
st.text("The assistant is based on https://dblp.org/ and https://arxiv.org/")

openai.api_key = st.secrets.openai_key

with st.sidebar:
    st.markdown("""
    # How does it work?
    Upload your documents and prompt your research topics
                
    Example of prompts:
    - prepare literature review on labour market and artificial intelligence
    - prepare literature review on gpt and virtual assistant
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),
                unsafe_allow_html=True)
    st.markdown(
        """
        <a href="https://www.linkedin.com/in/mauropelucchi" target="_blank">
        @ Mauro Pelucchi
        </a>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),
                unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True, type=['pdf'])
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            text_data = ""
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                for page in doc:
                    text_data += page.get_text()
            new_documents = [
                Document(
                    text=text_data,
                    metadata={},
                )
            ]
            if index == None:
                index = VectorStoreIndex.from_documents(new_documents, show_progress=False)
                st.session_state.chat_engine = index.as_chat_engine(
                    chat_mode="condense_question", verbose=True, streaming=True
                )
            else:
                index.insert_nodes(parser.get_nodes_from_documents(new_documents))

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
            temperature=0.3,
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
            The process to write literature review is the following:
            1) extract relevant terms
            2) download documents
            3) find 15 relevant papers
            4) write the literature review
            """,
)

if "chat_engine" not in st.session_state.keys() and index != None:  # Initialize the chat engine
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
    if index != None:
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
                    response_text = response_text + token
                parser = SimpleNodeParser()
                for response in response_text.split('####'):
                    st.text(f"Downloading new relevant documents about {response}...")
                    new_documents = get_academic_papers_from_dblp(response)
                    new_documents.extend(get_arxiv_documents(response))
                    st.text("Adding new docs to the existing index...")
                    index.insert_nodes(parser.get_nodes_from_documents(new_documents))
                st.text("Index is ready")
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response_stream.response_gen)
            message = {"role": "assistant", "content": response_stream.response}
            st.session_state.messages.append(message)
    else:
        st.text("Upload your documents to start the process")

