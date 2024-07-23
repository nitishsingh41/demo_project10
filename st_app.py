import streamlit as st
from io import BytesIO
from chatbot import process_query

# Title of the application
st.title("Candidate Chatbot")

# Sidebar configuration for source selection and file upload
st.sidebar.header("Upload Source")

# Source type selection
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

# Initialize chat history and source path in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "source_path" not in st.session_state:
    st.session_state.source_path = None

# Function to handle file uploads and start chat
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

# Start chat when the sidebar button is clicked
if st.sidebar.button("Upload and Start Chat"):
    handle_upload()
    if st.session_state.source_path:
        st.session_state.chat_active = True  # Activate chat

# Display chat messages if chat is active
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

        # Check if source_path is defined before using it
        if st.session_state.source_path is not None:
            # Process the query and get the response
            response = process_query(input=prompt, source_path=st.session_state.source_path, source_type=processed_source_type)['answer'].split(': ')[-1]
            # Display assistant response and add to chat history
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("No valid source available for processing.")
