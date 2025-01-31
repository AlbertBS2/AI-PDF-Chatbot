from dotenv import dotenv_values
from groq import Groq
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from io import BytesIO


############################ VARIABLES ############################

credentials = dotenv_values()

GROQ_API_KEY = credentials['GROQ_API_KEY']
client = Groq(api_key=GROQ_API_KEY)


############################ FUNCTIONS ############################

def get_ai_response(query, model, context=None, chat_history=None):
    """
    Send query to the LLM with context and chat history included.
    """
    history = ""
    if chat_history:
        for i, msg in enumerate(chat_history):
            role = "User" if i % 2 == 0 else "Assistant"
            history += f"{role}: {msg}\n"

    prompt = f"Conversation history: \n{history}\n\nContext: \n{context}\n\nQuestion: \n{query}"

    if model=="Llama 3.3":
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
        )

        response = chat_completion.choices[0].message.content + "\n"
    
    elif model=="DeepSeek":
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="deepseek-r1-distill-llama-70b",
        )

        response = chat_completion.choices[0].message.content + "\n"
        response = response.split("</think>\n\n", 1)[1]

    return response


def get_pdf_text(pdf_doc):
    """
    Extract raw text from PDFs.
    """
    raw_text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        raw_text += page.extract_text()
    return raw_text


def get_text_chunks(raw_text):
    """
    Split raw text into manageable chunks.
    """
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    chunks = text_splitter.split_text(raw_text)
    return chunks


def create_vector_store(text_chunks):
    """
    Create a vectorstore from text chunks.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


def find_relevant_chunks(query, vector_store, k=4):
    """
    Find the top-k most relevant chunks from the vectorstore.
    """
    relevant_chunks = vector_store.similarity_search(query, k=k)
    return "\n".join([chunk.page_content for chunk in relevant_chunks])


############################ MAIN ############################

def main():
    st.set_page_config(page_title='AI Chatbot', page_icon='🤖')

    st.title('AI Chatbot')

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Expandable PDF file uploader
    with st.expander("📄 Attach pdf file"):
        pdf_file = st.file_uploader('Upload your PDF file', type=['pdf'], accept_multiple_files=False)

    # Index PDF content
    with st.spinner('Indexing PDF content...'):
        if pdf_file is not None:
            raw_text = get_pdf_text(pdf_file)
            text_chunks = get_text_chunks(raw_text)
            vector_store = create_vector_store(text_chunks)

    # Model selection using a radio button
    model_choice = st.radio(
        "Choose a model:",
        ("Llama 3.3", "DeepSeek"),
        horizontal=True,
        index=0  # Default selection
    )

    # Display chat history as a conversation
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message('User'):
                st.write(f"**You**: {msg}")
        else:
            with st.chat_message('Assistant'):
                st.write(f"**Assistant**: {msg}")

    # Chat input
    input_query = st.chat_input("Ask me anything!")

    # Retrieve answer using the selected AI model
    if input_query:
        with st.spinner('Thinking...'):
            if pdf_file is not None:
                # Retrieve relevant context
                context = find_relevant_chunks(input_query, vector_store)
            else:
                context = None

            # Get AI response
            response = get_ai_response(input_query, model_choice,context, st.session_state.chat_history)

            # Update chat history
            st.session_state.chat_history.append(input_query)
            st.session_state.chat_history.append(response)

            # Display the new messages
            with st.chat_message("user"):
                st.write(f"**You:** {input_query}")
            with st.chat_message("assistant"):
                st.write(f"**Assistant:** {response}")

    # Start a new chat and clear history
    if st.session_state.chat_history:
        if st.button("New chat", icon="💬"):
            st.session_state.chat_history = []
            st.rerun()


if __name__ == '__main__':
    main()
