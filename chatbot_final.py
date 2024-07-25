import os
import sys

# Import necessary libraries from LangChain and related modules
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

# Workaround for sqlite3
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


class EmbeddingModel:
    def __init__(self, model_id: str, device: str):
        self.model = HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs={'device': device, 'trust_remote_code': True},
            encode_kwargs={'device': device},
        )


class DocumentIngestor:
    @staticmethod
    def ingest(source_type: str, source_path: str):
        """
        Ingest personal data from various sources using LangChain.
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


class RAGChatbot:
    def __init__(self, llm, embed_model):
        self.llm = llm
        self.embed_model = embed_model
        self.current_source_path = None
        self.vectorstore = None
        self.history_aware_retriever = None
        self.question_answer_chain = None
        self.conversational_rag_chain = None
        self.store = {}

    def initialize_components(self, input: str, session_id: str, source_path: str, source_type: str):
        # Reset components if a new document is provided
        if source_path != self.current_source_path:
            print("Resetting components for new document:", source_path)

            # Cleanup old Chroma client if it exists
            if self.vectorstore is not None:
                self.vectorstore.delete_collection()
                del self.vectorstore  # Remove the old vectorstore

            self.current_source_path = source_path

            # Ingest data and initialize vectorstore
            data = DocumentIngestor.ingest(source_type, source_path)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=25)
            all_splits = text_splitter.split_documents(data)

            # Initialize new vectorstore
            self.vectorstore = Chroma.from_documents(documents=all_splits, embedding=self.embed_model)

            # Initialize retriever
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

            # Initialize history-aware retriever
            self.history_aware_retriever = self.create_history_aware_retriever(retriever)

            # Initialize question-answer chain
            self.question_answer_chain = self.create_question_answer_chain()

            # Initialize conversational RAG chain
            self.conversational_rag_chain = self.create_conversational_rag_chain(session_id)

        else:
            print("Using existing components for document link:", source_path)

    def create_history_aware_retriever(self, retriever):
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

        return create_history_aware_retriever(self.llm, retriever, contextualize_q_prompt)

    def create_question_answer_chain(self):
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Keep the answer concise. Don't ask follow-up questions."
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

        return create_stuff_documents_chain(self.llm, qa_prompt)

    def create_conversational_rag_chain(self, session_id: str):
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]

        return RunnableWithMessageHistory(
            self.create_retrieval_chain(),
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def create_retrieval_chain(self):
        return create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

    def process_query(self, input: str, session_id='abc123', source_path='./resume.pdf', source_type='pdf'):
        if not source_path:
            return {"answer": "Please provide document path"}

        self.initialize_components(input, session_id, source_path, source_type)

        response = self.conversational_rag_chain.invoke(
            {"input": input},
            config={"configurable": {"session_id": session_id}}
        )

        return {"answer": response["answer"]}


# Main execution
if __name__ == "__main__":
    pass
