from dotenv import load_dotenv
import streamlit as st
import openai
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask me about Raj Ganesh")
    st.markdown("This Chatbot is developed for showcasing the AI Skillset and ask any other Questions regarding his hiring process - Developed by Raj Ganesh ")
    
    # upload file
    pdf = Path("raj.pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      user_question = st.text_input("Ask a question")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)
    
    link1 = '[Schedule a Interview](https://calendly.com/rajganesh/30min)'
    st.markdown(link1, unsafe_allow_html=True)
    link2 = '[Download Resume](https://typewriter.ae/raj_resume.pdf)'
    st.markdown(link2, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
