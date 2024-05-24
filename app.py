#nessesory liberaries and framewroks
import os
import streamlit as st
from langchain_community.llms import ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# Initialize the language model with the specified model name
llm = ollama.Ollama(model="llama3:8b") 

def main():

    """Sets up the Streamlit app and handles file upload and chat interaction."""

    # Initialize session state variables if they don't exist
    st.set_page_config(page_title="Chat with your file")
    st.header("Chat with your document")

    # Initialize session state variables if they don't exist
    if 'processComplete' not in st.session_state:
        st.session_state['processComplete'] = None
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Sidebar for file upload
    with st.sidebar:
        uploaded_files =  st.file_uploader(
            "Upload your file",
            type=['pdf'],
            accept_multiple_files=True
        )
        process = st.button("Process")

    # Process the uploaded files
    if process:

        st.write("The file has being processing")
        files_text = get_files_text(uploaded_files)
        text_chunks = get_chunks(files_text)

        # Create vector store for the text chunks 
        vectorstore = get_vectorstore(text_chunks)
    
        st.session_state.conversation = vectorstore
        st.session_state.processComplete = True
        st.write("You are ready for chat")


    if st.session_state.processComplete == True:

        user_question = st.chat_input("Ask Question about your files.")
        if user_question:   
            handle_user_input(user_question)

    display_response()

# Function to get the input file and read the text from it.
def get_files_text(uploaded_files):

    '''Extracts text from uploaded PDF files.'''
    
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1] 

        # Save the uploaded file temporarily
        if file_extension == ".pdf":
           with open(split_tup[0], "wb") as f:
            f.write(uploaded_file.read())

            # Load and split the PDF file into pages
           loader  = PyPDFLoader(split_tup[0])
           page = loader.load_and_split() 

    return page

def get_chunks(pages):

    '''Splits text into manageable chunks.'''

    text_splitter = CharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
    )

    # Split the documents into chunks
    chunks = text_splitter.split_documents(pages)
    return chunks


def get_vectorstore(pages):

    '''Creates a vector store from the text chunks.'''

    #Create emdedings by using huggingface emdedding model
    embeddings  = HuggingFaceEmbeddings()
    embeddings.model_name = "sentence-transformers/all-mpnet-base-v2"

     # Create documents from the text chunks
    documents = [
        Document(page_content=page.page_content,
        metadata=page.metadata) for page in pages
        ]

    # Create a vector store using FAISS
    db = FAISS.from_documents(documents, embeddings)
    return db

def handle_user_input(user_question):
    
    """Handles user input and generates a response using the LLM."""

    # Retrieve the vector store
    vectorstore =  st.session_state.conversation
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
        )

    # Retrieve relevant documents based on the user's question
    context_documents = retriever.invoke(user_question)
    formatted_context = "\n".join([doc.page_content for doc in context_documents])

    # Define the message template
    message_template = """
    Answer this question using the provided context only.

    Question:
    {user_question}

    File content:
    {formatted_context}
    if the question is not relivent to the file content then 
    told the user please asked relivent question to your doucumnet 
    """

    # Create the ChatPromptTemplate with the message template
    prompt_template = ChatPromptTemplate.from_template(message_template)

    # Render the prompt with specific values
    user_question_with_context = prompt_template.format(
        user_question=user_question,
        formatted_context=formatted_context
        )


    # Generate a response using the LLM
    response = llm.invoke(user_question_with_context)

    # Update the chat history in session state
    st.session_state.chat_history.append({"user": user_question, "assistant": response})
    return response


def display_response():

    """show the response on web page"""

    # Display the chat history
    response_container = st.container()
    with response_container:
        for messages in st.session_state.chat_history:
            with st.chat_message("User"):
                st.markdown(f"**User:** {messages['user']}")
            with st.chat_message("Assistant"):
                st.markdown(f"**assistant:** {messages['assistant']}")


if __name__ == '__main__':
    main()






