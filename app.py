import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI 
from htmlTemplates import css,bot_template,user_template
import os

def get_pdf_text(pdf_list):
    text=""
    for pdf in pdf_list:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

# Extract text from multiple PDFs
def extract_text_from_pdf(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:  # Iterate over the list of PDF paths
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"  # Add newline to separate text from different PDFs
        except FileNotFoundError:
            text += f"Error: The file '{pdf_path}' was not found.\n"
    return text

def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        st.warning("Please upload the textual PDF file - this is PDF files of image")
        return None
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain


def handle_userInput(user_question):
    response=st.session_state.conversation({'question':user_question})
    st.session_state.chat_history=response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",msg.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",msg.content),unsafe_allow_html=True)

def main():
    load_dotenv()

    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    st.set_page_config(page_title="PDF's Chat Agent")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "train" not in st.session_state:
        st.session_state.train = False

    st.header("Multi-PDF's :books: - Chat Agent :robot_face:")

    with st.sidebar:
        st.subheader(":file_folder: PDF File's Section")
        pdf_list = st.file_uploader("Upload your PDF files here and train agent", type=['pdf'], accept_multiple_files=True)
        train = st.button("Train the Agent")
        if train:
            with st.spinner("Processing"):
                # 1 - Get the text from PDFs
                raw_text = get_pdf_text(pdf_list)
                
                # 2 - get the text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # 3 - Create vector store
                vector_store = get_vector_store(text_chunks)
                
                # 4 - conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)
                
                # set train to True to indicate agent has been trained
                st.session_state.train = True

    if not st.session_state.train:
        st.warning("First Train the Agent")

    if st.session_state.train:
        st.write("<h5><br>Ask anything from your documents:</h5>", unsafe_allow_html=True)
        user_question = st.text_input(label="", placeholder="Enter something...")
        if user_question:
            handle_userInput(user_question)

if __name__ == "__main__":
    main()
