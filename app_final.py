import streamlit as st
from io import BytesIO
from chatbot_final import RAGChatbot, EmbeddingModel
from langchain_groq import ChatGroq

## Title of the application
st.title("Candidate Chatbot")

## Sidebar configuration for API key and source selection
st.sidebar.header("Configuration")

# Ask for API key first
if "api_key" not in st.session_state:
    api_key = st.sidebar.text_input("Enter GROQ API Key:", type="password")
    
    # Store the API key in session state if provided
    if api_key:
        st.session_state.api_key = api_key  # Store API key
        st.success("API Key stored successfully.")
    else:
        st.warning("Please enter your API key to proceed.")
else:
    st.success("API Key is already set.")

# Source type selection only if API key is set
if "api_key" in st.session_state:
    st.sidebar.header("Upload Source")

    source_type = st.sidebar.selectbox("Select Source Type", options=['pdf', 'txt', 'website'])

    # Initialize variables for uploaded files and URL
    uploaded_pdf = None
    uploaded_txt = None
    web_url = None

    # File upload or URL input based on selected source type
    if source_type == 'pdf':
        uploaded_pdf = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    elif source_type == 'txt':
        uploaded_txt = st.sidebar.file_uploader("Upload a TXT file", type=["txt"])
    else:  # website
        web_url = st.sidebar.text_input("Enter Website URL:", value="https://example.com")

    ## Initialize chat history and source path in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "source_path" not in st.session_state:
        st.session_state.source_path = None
    if "rag_chatbot" not in st.session_state:
        # Load embedding model
        embed_model_id = "Alibaba-NLP/gte-large-en-v1.5"
        device = 'cuda'
        embed_model = EmbeddingModel(embed_model_id, device)

        # Create RAG chatbot instance
        llm = ChatGroq(
            temperature=0,
            model="llama3-70b-8192",
            max_tokens=4000,
            api_key=st.session_state.api_key  # Use the stored API key
        )
        st.session_state.rag_chatbot = RAGChatbot(llm, embed_model)

    ## Function to handle file uploads and start chat
    def handle_upload():
        if source_type == 'pdf' and uploaded_pdf is not None:
            with open("uploaded_document.pdf", "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            st.success("PDF uploaded successfully.")
            st.session_state.source_path = "uploaded_document.pdf"
        elif source_type == 'txt' and uploaded_txt is not None:
            with open("uploaded_document.txt", "wb") as f:
                f.write(uploaded_txt.getbuffer())
            st.success("TXT file uploaded successfully.")
            st.session_state.source_path = "uploaded_document.txt"
        elif source_type == 'website' and web_url:
            st.session_state.source_path = web_url
            st.success("Website URL accepted.")
        else:
            st.error(f"Please upload a valid {source_type} file or enter a valid website URL.")

    ## Start chat when the sidebar button is clicked
    if st.sidebar.button("Upload and Start Chat"):
        handle_upload()

        # Activate chat
        st.session_state.chat_active = True

## Display chat messages if chat is active
if "chat_active" in st.session_state and st.session_state.chat_active:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input for chat
    if prompt := st.chat_input("What can I help you with?"):
        # Display user message and add to chat history
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Prepare source type for processing
        processed_source_type = 'text' if source_type == 'txt' else source_type

        # Check if source_path is defined and rag_chatbot is initialized
        if st.session_state.source_path is not None and 'rag_chatbot' in st.session_state:
            # Process the query using RAGChatbot
            query_response = st.session_state.rag_chatbot.process_query(
                prompt,
                session_id='abc123',  # You can replace 'abc123' with a dynamic session ID if needed
                source_path=st.session_state.source_path,
                source_type=processed_source_type
            )
            # Display assistant response and add to chat history
            with st.chat_message("assistant"):
                st.markdown(query_response)
            st.session_state.messages.append({"role": "assistant", "content": query_response})
        else:
            st.error("No valid source available for processing or API key is missing.")
