from pathlib import Path
from dotenv import dotenv_values
from groq import Groq
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


############################ VARIABLES ############################

credentials = dotenv_values()

GROQ_API_KEY = credentials['GROQ_API_KEY']
client = Groq(api_key=GROQ_API_KEY)

project_dir = Path(__file__).parent
data_folder = project_dir / "data"


############################ FUNCTIONS ############################

def get_ai_response(query, context=None, chat_history=None):
    """
    Send query to the LLM with context and chat history included.
    """
    history = ""
    if chat_history:
        for i, msg in enumerate(chat_history):
            role = "User" if i % 2 == 0 else "Assistant"
            history += f"{role}: {msg}\n"

    #prompt = f"Context: {context}\n\nQuestion: {query}"
    prompt = f"Conversation history: \n{history}\n\nContext: \n{context}\n\nQuestion: \n{query}"

    chat_completion = client.chat.completions.create(
        messages=[
             {
                 "role": "user",
                 "content": prompt,
             }
        ],
        #model="llama-3.3-70b-versatile",
        model="deepseek-r1-distill-llama-70b",
    )

    response = chat_completion.choices[0].message.content + "\n"
    return response


def get_pdf_text(pdf_docs):
    """
    Extract raw text from PDFs.
    """
    raw_text = ""
    for pdf_doc in pdf_docs:
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

docs = [str(file_path) for file_path in data_folder.iterdir()]
raw_text = get_pdf_text(docs)
text_chunks = get_text_chunks(raw_text)
vector_store = create_vector_store(text_chunks)

print("PDF content successfully indexed. You can now ask questions!\n")

chat_history = [] # Initialize conversation history

input_query = input("Ask me anything! Type exit to end the conversation: \n")

while str.lower(input_query) != "exit":
    # Retrieve relevant context
    context = find_relevant_chunks(input_query, vector_store)

    # Get AI response with context
    response = get_ai_response(input_query, context, chat_history)

    # Update chat history
    chat_history.append(input_query)
    chat_history.append(response)

    print(response)
    input_query = input("Ask me anything! Type exit to end the conversation: \n")
