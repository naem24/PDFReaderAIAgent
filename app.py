# Imports go here
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from htmlTemplates import css,bot_template,user_template
from PyPDF2 import PdfReader
import os

# Start Helper functions 

# Extract text from the PDF
def get_pdf_text(pdf_list):
    text=""
    for pdf in pdf_list:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to answer questions
def answer_question(pdf_text, question):
    if not pdf_text:
        return "Error: No text extracted from the PDFs."
    
    answer = qa_chain.invoke({"context": pdf_text, "question": question})  # Updated to use invoke
    return_text = answer.content if hasattr(answer, 'content') else answer  # Handle response content

    st.write(bot_template.replace("{{MSG}}",return_text),unsafe_allow_html=True)

# End helper functions

def main():
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    st.set_page_config(page_title="PDF's Agent")
    st.write(css, unsafe_allow_html=True)

    if "setup" not in st.session_state:
        st.session_state.setup = False

    st.header("PDF Agent")

    with st.sidebar:
        st.subheader(":file_folder: PDF File's Section")
        
        pdf_paths = st.file_uploader("Upload your PDF files here and train agent", type=['pdf'], accept_multiple_files=True)
        
        setup = st.button("Setup the Agent")

        # If user selects to Setup, go ahead
        if setup:
            with st.spinner("Setting up..."):
                # 1 - Get the text from PDFs
                pdf_text = get_pdf_text(pdf_paths)
                
                # 2 - Define the prompt template
                # Define prompt template
                template = """
                You are an expert AI assistant. Use the information provided for answering the question
                Context: {context}
                Question: {question}
                Answer:
                """
                prompt = PromptTemplate(input_variables=["context", "question"], template=template)
                
                # 3 - Initialize Gemini LLM and chain
                llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=os.environ["GOOGLE_API_KEY"])
                qa_chain = RunnableSequence(prompt | llm)  # Updated to use RunnableSequence
                
                # Set setup to True to indicate agent is ready
                st.session_state.setup = True

    if not st.session_state.setup:
        st.warning("First setup the Agent")

    if st.session_state.setup:
        st.write("<h5><br>Ask anything from your documents:</h5>", unsafe_allow_html=True)
        user_question = st.text_input(label="", placeholder="Enter your query...")
        
        if user_question:
            answer_question(pdf_text, user_question)
 
if __name__ == "__main__":
    main()

