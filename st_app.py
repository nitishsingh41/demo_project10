import streamlit as st
import requests
from io import BytesIO
from app import process_query

st.title("PDF/TXT/Website Chat Bot")

# Create a sidebar for source selection and file upload
st.sidebar.header("Upload Source")

# Select source type in the sidebar
source_type = st.sidebar.selectbox("Select Source Type", options=['pdf', 'txt', 'website'])

# Initialize variables for uploaded files and URL
uploaded_pdf = None
uploaded_txt = None
web_url = None

# Input based on selected source type in the sidebar
if source_type == 'pdf':
    uploaded_pdf = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
elif source_type == 'txt':
    uploaded_txt = st.sidebar.file_uploader("Upload a TXT file", type=["txt"])
else:  # website
    web_url = st.sidebar.text_input("Enter Website URL:", value="https://example.com")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process the upload and start chat when the sidebar button is clicked
if st.sidebar.button("Upload and Start Chat"):
    if source_type == 'pdf':
        if uploaded_pdf is not None:
            with open("uploaded_document.pdf", "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            st.success("PDF uploaded successfully.")
            source_path = "uploaded_document.pdf"
            st.session_state.chat_active = True  # Activate chat
        else:
            st.error("Please upload a PDF file.")
            source_path = None
    elif source_type == 'txt':
        if uploaded_txt is not None:
            with open("uploaded_document.txt", "wb") as f:
                f.write(uploaded_txt.getbuffer())
            st.success("TXT file uploaded successfully.")
            source_path = "uploaded_document.txt"
            st.session_state.chat_active = True  # Activate chat
        else:
            st.error("Please upload a TXT file.")
            source_path = None
    else:  # website
        if web_url:
            source_path = web_url
            st.success("Website URL accepted.")
            st.session_state.chat_active = True  # Activate chat
        else:
            st.error("Please enter a valid website URL.")
            source_path = None

# Display chat messages from history if chat is active
if "chat_active" in st.session_state and st.session_state.chat_active:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What can I help you with?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        if source_type=='txt':
          source_type='text'
        # Simulate a response (replace with actual processing logic)
        response = process_query(input=prompt,source_path=source_path, source_type=source_type)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
