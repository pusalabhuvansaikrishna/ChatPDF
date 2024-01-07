import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


 
def get_pdf_text(pdf_files):
  text=""
  for pdf in pdf_files:
    pdf_reader=PdfReader(pdf)
    for page in pdf_reader.pages:
      text+=page.extract_text()
  return text

def get_text_chunks(raw_text):
  text_splitter=CharacterTextSplitter(separator='\n',chunk_size=1000,chunk_overlap=250,length_function=len)
  chunks=text_splitter.split_text(raw_text)
  return chunks

def get_embeddings(text_chunks):
  embeddings=OpenAIEmbeddings()
  vectorstore=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
  return vectorstore

def get_conversation_chain(vectors):
  llm=ChatOpenAI()
  memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
  conversation_chain=ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectors.as_retriever(),memory=memory)
  return conversation_chain

def handle_userinput(query):
  response=st.session_state.conversation({'question':query})
  st.session_state.chat_history=response['chat_history']
  for i,message in enumerate(st.session_state.chat_history):
    if i%2==0:
      with st.chat_message(name='User',avatar='✨'):
        st.write(message.content)
    else:
      with st.chat_message(name='Assitant',avatar="🦾"):
        st.write(message.content)


def main():
  load_dotenv()
  st.set_page_config(page_title="ChatPDF",page_icon='📚')
  if "conversation" not in st.session_state:
    st.session_state.conversation=None
  
  if "chat_history" not in st.session_state:
    st.session_state.chat_history=None
  
  st.title("Chat with Your PDFs")
  query=st.text_input("Enter your Query Here")
  if query:
    handle_userinput(query)

  with st.sidebar:
    st.title("Documents Here")
    pdf_files=st.file_uploader("Upload your PDF here to begin chat!",type='pdf',accept_multiple_files=True)
    if(st.button("Process🚀")):
      with st.spinner("Processing..."):
        raw_text=get_pdf_text(pdf_files)
        text_chunks=get_text_chunks(raw_text)
        vectors=get_embeddings(text_chunks)
        st.session_state.conversation=get_conversation_chain(vectors)
        st.success("Done")
  #st.session_state.conversation


main()