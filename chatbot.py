import os
import sys

# Import necessary libraries from LangChain and related modules
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Workaround for sqlite3
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain_chroma import Chroma

# Load embedding model
embed_model_id = "Alibaba-NLP/gte-large-en-v1.5"
device = 'cuda'
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device, 'trust_remote_code': True},
    encode_kwargs={'device': device},
)

# Download Llama2 7B model if not already downloaded
if not os.path.exists('./llm.gguf'):
    #os.system("wget -q https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q8_0.gguf?download=true -O llm.gguf")
    #os.system("wget -q https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf?download=true -O llm.gguf")
    #os.system("wget -q https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q8_0.gguf?download=true -O llm.gguf")
    #os.system("wget -q https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q8_0.gguf?download=true -O llm.gguf")
    os.system("wget -q https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q8_0.gguf?download=true -O llm.gguf")
# Initialize LLM model using LlamaCpp
llm = LlamaCpp(
    model_path="./llm.gguf",
    n_gpu_layers=-1,
    n_ctx=4000,
    temperature=0.1,
    verbose=False,
    max_tokens=100
)

# Function to ingest personal data from various sources
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
    elif source_type == 'pdf':
        loader = PyPDFLoader(source_path)
    elif source_type == 'website':
        loader = WebBaseLoader(source_path)
    else:
        raise ValueError("Invalid source_type. Choose 'text', 'pdf', or 'website'.")

    return loader.load()

# API initialization variables
web_page_link_old = ""
history_aware_retriever = None
question_answer_chain = None
rag_chain = None
store = {}
conversational_rag_chain = None

# Function to initialize components for the RAG chatbot
def initialize_components(input: str, session_id: str, source_path: str, source_type: str):
    global web_page_link_old, vectorstore, history_aware_retriever, question_answer_chain, rag_chain, conversational_rag_chain, store
    
    web_page_link = source_path
    
    if web_page_link != web_page_link_old:
        print("Initializing components for document link:", web_page_link)
        web_page_link_old = web_page_link
        
        # Ingest data and initialize vectorstore
        data = ingest_personal_data(source_type, source_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=25)
        all_splits = text_splitter.split_documents(data)

        # Initialize or reset vectorstore
        try:
            vectorstore.delete_collection()
            del vectorstore
        except:
            pass
        
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)
        
        # Initialize retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # Initialize history-aware retriever
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
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

        # Initialize question-answer chain
        system_prompt = (
            "You are an assistant which gives precise answers. "
            "Strictly use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Keep the answer concise and very short and don't provide extra details. "
            #"Don't make additional queries or generate unsupported inferences."
            #"Please provide information without any follow-up questions."
            "Don't generate any follow-up questions."
            "\n\n{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Initialize RAG chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Initialize store for session history
        store = {}
        
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]
        
        # Initialize conversational RAG chain
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    else:
        print("Using existing components for web page link:", web_page_link)

# Function to process user queries
def process_query(input: str, session_id='abc123', source_path='./resume.pdf', source_type='pdf'):
    if source_path == "":
        return {"answer": "Please provide document path"}
    
    initialize_components(input, session_id, source_path, source_type)
    
    response = conversational_rag_chain.invoke(
        {"input": input},
        config={
            "configurable": {"session_id": session_id}
        }
    )

    return {"answer": response["answer"]}

# Main execution
if __name__ == "__main__":
    pass
