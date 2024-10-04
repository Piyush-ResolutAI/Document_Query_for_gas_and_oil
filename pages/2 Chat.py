import pysqlite3,sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
# from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_groq import ChatGroq
from audiorecorder import audiorecorder
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from groq import Groq
import tempfile
import numpy as np
import av
import os

load_dotenv()

# class AudioProcessor(AudioProcessorBase):
#     def __init__(self):
#         self.audio_data = []

#     def recv(self, frame: av.AudioFrame):
#         audio = frame.to_ndarray()
#         self.audio_data.append(audio)
#         return frame

#     def get_audio_data(self):
#         if len(self.audio_data) > 0:
#             return np.concatenate(self.audio_data)
#         return None


def get_vectorstore(domain):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory=f"{domain}", embedding_function=embeddings)
    
    return vectorstore

def get_context_retriever_chain(vector_store):
    # llm = ChatGroq(model="llama-3.1-70b-versatile", temperature = 0)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    # llm = ChatGroq(model="llama-3.1-70b-versatile", temperature = 0)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    # st.sidebar.write(response["context"])
    
    filepaths = set()
    try:
        for document in response["context"]:
            # st.sidebar.write(document)
            metadata = document.metadata
            filepath = metadata["source"]
            document = filepath.split("\\")[-1]
            # st.sidebar.write(document)
            filepaths.add(document)
        # st.sidebar.write(filepaths)
    except:
        pass
    
    return response['answer'], filepaths



st.set_page_config(page_title='chat with your Documents', page_icon=':books:')
st.image("LOGO.png", width=455)
st.header("Start Chatting with your Documents")
client = Groq()
# try:
if st.session_state.domains:
    
    domain = st.sidebar.selectbox("Select a Domain", options=st.session_state.domains)
    
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    # if "vector_store" not in st.session_state:
    vector_store = get_vectorstore(domain)
    
    with st.sidebar:
        st.subheader("Click below to record")
        audio = audiorecorder("Click to record", "Click to stop recording")
    
    if len(audio) > 0:
        audio.export("audio.wav", format="wav")
        filename = "audio.wav"
        with open(filename, "rb") as file:
            # Create a transcription of the audio file
            transcription = client.audio.transcriptions.create(
            file=(filename, file.read()), # Required audio file
            model="distil-whisper-large-v3-en", # Required model to use for transcription
            prompt="Specify context or spelling",  # Optional
            response_format="json",  # Optional
            language="en",  # Optional
            temperature=0.0  # Optional
            )
            # Print the transcription text
            print(transcription.text)
            st.write(f"you said: {transcription.text}")
            response, file_paths = get_response(transcription.text)
            st.sidebar.subheader("Below are the Source Documents:")
            for file_path in file_paths:
                st.sidebar.info(file_path)
            st.session_state.chat_history.append(HumanMessage(content = transcription.text))
            st.session_state.chat_history.append(AIMessage(content = response))
    
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response, file_paths = get_response(user_query)  ##Response
        st.sidebar.subheader("Below are the Source Documents:")
        for file_path in file_paths:
            st.sidebar.info(file_path)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
                    
        
# except:
#     st.warning("Please login first")