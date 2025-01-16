from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import json
import os

load_dotenv()

model = os.getenv('LLM_MODEL', 'meta-llama/Meta-Llama-3.1-405B-Instruct')
rag_directory = os.getenv('DIRECTORY', 'books')

# Function to cache the local model for better performance
@st.cache_resource
def get_local_model():
    return HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        max_new_tokens=1024,
        do_sample=False
    )

# Getting the local model
llm = get_local_model()

# Function to load the documents & split them into smaller chunks
def load_documents(directory):
    # Load the PDF or txt documents from the directory
    loader = DirectoryLoader(directory)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    return docs

# Function to convert chunks into vector embeddings & store in a vector database
@st.cache_resource
def get_chroma_instance():
    # Get the documents split into chunks
    docs = load_documents(rag_directory)

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load it into Chroma
    return Chroma.from_documents(docs, embedding_function)

# Getting the cached db
db = get_chroma_instance()  

# Function to query the db using similarity algorithms for info to answer user questions 
def query_documents(question):
    """
    Uses RAG to query documents for information to answer a question

    Example call:

    query_documents("Can you give me some key points written in the book X?")
    Args:
        question (str): The question the user asked that might be answerable from the searchable documents
    Returns:
        str: The list of texts (and their sources) that matched with the question the closest using RAG
    """
    similar_docs = db.similarity_search(question, k=5)
    docs_formatted = list(map(lambda doc: f"Source: {doc.metadata.get('source', 'NA')}\nContent: {doc.page_content}", similar_docs))

    return docs_formatted   

def prompt_ai(messages):
    # Fetch the relevant documents for the query
    user_prompt = messages[-1].content
    retrieved_context = query_documents(user_prompt)
    formatted_prompt = f"Context for answering the question:\n{retrieved_context}\nQuestion/user input:\n{user_prompt}"    

    # Prompt the AI with the latest user message
    doc_chatbot = ChatHuggingFace(llm=llm)
    ai_response = doc_chatbot.invoke(messages[:-1] + [HumanMessage(content=formatted_prompt)])

    return ai_response

def main():
    st.title("Chat with Local Documents")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=f"You are a personal assistant who answers questions based on the context provided if the provided context can answer the question. You only provide the answer to the question/user input and nothing else. The current date is: {datetime.now().date()}")
        ]    

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        message_json = json.loads(message.model_dump_json())
        message_type = message_json["type"]
        if message_type in ["human", "ai", "system"]:
            with st.chat_message(message_type):
                st.markdown(message_json["content"])        

    # React to user input
    if prompt := st.chat_input("What will you like to know?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            ai_response = prompt_ai(st.session_state.messages)
            st.markdown(ai_response.content)
        
        st.session_state.messages.append(ai_response)

if __name__ == "__main__":
    main()