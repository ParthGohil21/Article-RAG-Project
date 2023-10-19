import streamlit as st
import pickle
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title='Article RAG Chatbot ', page_icon='📜')
st.title('Article Retrieval Augmented Generation - RAG Chatbot 📜')
st.sidebar.title('Article URLs')
n=st.sidebar.slider('How many URLs', 1 , 10)
placeholder=st.empty()
file_path='faiss_vectors'

llm=OpenAI(temperature=0.5)

def main():
    
    urls=[]
    for i in range(n):
        url=st.sidebar.text_input(f'URL {i+1}')
        urls.append(url)

    process=st.sidebar.button('Process')
    if process:
        raw_data= load_data(urls)

        text_chunks=get_text_chunks(raw_data)
        
        vector_stores=vector_embeddings(text_chunks)
        with open(file_path, "wb") as f:
            pickle.dump(vector_stores, f)
        
    query = placeholder.text_input("Ask a question:")
    if query:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vector_stores = pickle.load(f)
                
                chain=RetrievalQAWithSourcesChain.from_llm(
                    llm=llm,
                    retriever=vector_stores.as_retriever()
                )

                result=chain({'question': query})
                st.header('Answer')
                st.write(result['answer'])

                sources=result.get('sources', '')
                if sources:
                    st.subheader('Sources')
                    source_list=sources.split('\n')
                    for source in source_list:
                        st.write(source)


def load_data(urls):
    loader=UnstructuredURLLoader(urls=urls)
    placeholder.text('Data Loading Started...✅✅✅')
    data=loader.load()
    return data
    

def get_text_chunks(data):
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.'],
        chunk_size=500,
        chunk_overlap=150
    )
    
    text_chunks=text_splitter.split_documents(data)
    return text_chunks

def vector_embeddings(text_chunks):
    placeholder.text('Vector Embeddings Started...✅✅✅')
    embeddings=HuggingFaceBgeEmbeddings()
    vector_stores=FAISS.from_documents(text_chunks, embeddings)
    return vector_stores
    

if __name__ == '__main__':
    main()
