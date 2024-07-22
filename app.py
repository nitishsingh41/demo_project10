import os
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.document_loaders import TextLoader, PDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def ingest_personal_data(source_type, source_path):
    """
    Ingest personal data from various sources using LangChain.
    
    Args:
        source_type (str): Type of data source ('text', 'pdf', or 'website')
        source_path (str): Path to the data source (file path or URL)
    
    Returns:
        list: List of documents containing the ingested data
    """
    if source_type == 'text':
        loader = TextLoader(source_path)
        data = loader.load()

    
    elif source_type == 'pdf':
        loader = PDFLoader(source_path)
        data = loader.load()

    
    elif source_type == 'website':
        loader = WebBaseLoader(source_path)
        data = loader.load()
    
    else:
        raise ValueError("Invalid source_type. Choose 'text', 'pdf', or 'website'.")
    
    return data



#api initialize fastapi app and variables
#app = FastAPI()
web_page_link_old = ""  
history_aware_retriever = None
question_answer_chain = None
rag_chain = None
store = {}
conversational_rag_chain = None



#funtion to index webpage and make chat history based RAG chatbot
def initialize_components(input: str, session_id: str, source_path: str, source_type: str):
    global web_page_link_old,vectorstore, history_aware_retriever, question_answer_chain, rag_chain, conversational_rag_chain, store
    
    web_page_link = source_path
    
    if web_page_link != web_page_link_old:
        print("Initializing components for document link:", web_page_link)
        web_page_link_old = web_page_link
        
        # Initialize vectorstore
        data=ingest_personal_data(source_type, source_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        all_splits = text_splitter.split_documents(data)
        
        try:
            vectorstore.delete_collection()
            del vectorstore
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)
        except:
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)
        
        
        # Initialize retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # Initialize history_aware_retriever
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # Initialize question_answer_chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Strictly use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise and short."
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Initialize rag_chain which uses history_aware_retriever to answer the question
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Initialize store
        store = {}
        
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]
        
        # Initialize conversational_rag_chain
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    else:
        print("Using existing components for web page link:", web_page_link)
    
#funtion to run with input
def process_query(input: str, session_id='abc123', source_path:'./resume.pdf', source_type='pdf'):
    if source_path == "":
        return {"answer": "please provide document path"}
    
    initialize_components(input,session_id,source_path,source_type)
    
    response = conversational_rag_chain.invoke(
        {"input": input},
        config={
            "configurable": {"session_id":session_id}
        }
    )

    print("Response:", response)
    return {"answer": response["answer"]}

