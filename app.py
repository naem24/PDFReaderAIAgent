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

# Set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="PDF AI Agent",
    layout='wide'
)

# Hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
    #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
    div.block-container{padding-top:5rem;}
    div.stButton {text-align:center;}
    </style>
"""

# Hide the CSS code from the screen as they are embedded in markdown text. 
# Also, allow streamlit to unsafely process as HTML
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# This ensure we have slotted the main page in 3 columns - with the middle column taking 80% of the space 
with st.columns([0.10, 0.80, 0.10])[1]:
    pdf_paths = st.file_uploader("Upload your PDF files here to train the agent", type=['pdf'], accept_multiple_files=True)
    
    # 1 - Get the text from PDFs
    pdf_text = get_pdf_text(pdf_paths)
    
    # 2 - Define the prompt template
    # Define prompt template
    template = """
    You are an expert AI assistant. Use the information provided for answering the question. If the information is not available, 
    return 'Not Available' as an answer.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    
    # 3 - Initialize Gemini LLM and chain
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"])
    qa_chain = RunnableSequence(prompt | llm)  # Updated to use RunnableSequence

    # Define the options for the country dropdown
    options_list = ['Croatia', 'Cyprus', 'Denmark', 'Finland', 'France']

    # Create the selectbox
    selected_option = st.selectbox(
        'Select the country for which you want to view the SSI details', # Label
        options_list                         # Options
    )

    # Add the button to initiate the search
    fetch_selected = st.button('Get SSI Details')
    
    if fetch_selected:
        if not pdf_text:
            st.warning("Please upload the textual PDF file first")
        elif not selected_option:
            st.warning("Please select a country from the list first")
        else:
            with st.spinner("Fetching..."):
                user_question = "Show the Account number, place of settlement and SWIFT Code for the country " + selected_option
                
                answer = qa_chain.invoke({"context": pdf_text, "question": user_question})  # Updated to use invoke
                return_text = answer.content if hasattr(answer, 'content') else answer  # Handle response content
                st.write(return_text ,unsafe_allow_html=True)
