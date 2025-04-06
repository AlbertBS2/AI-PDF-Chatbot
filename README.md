# 📄🤖 AI-Powered PDF Chatbot

An interactive AI chatbot that allows users to upload PDF files and ask questions about their contents using powerful large language models (LLMs). This project leverages **LangChain**, **FAISS**, and **Hugging Face Embeddings** to extract, chunk, and vectorize document content for semantic search and contextual chat.

## 🔍 Features

- 📄 **PDF Uploading**: Upload one or more PDF files for document-based Q&A.
- 🧠 **Context-Aware Responses**: Retrieves the most relevant chunks from documents to generate informed answers.
- 🤖 **LLM Integration**: Choose between **Llama 3.3** and **DeepSeek** for dynamic AI-powered responses via the **Groq API**.
- 💬 **Chat Interface**: Friendly, session-based chat with persistent conversation history.
- 🧰 **Embeddings & Vector Store**: Uses `HuggingFaceEmbeddings` with `FAISS` to enable efficient similarity search.

## 🚀 Tech Stack

- Python
- Streamlit
- Groq API (LLMs: Llama 3.3, DeepSeek)
- LangChain
- FAISS
- Hugging Face Transformers
- PyPDF2
- python-dotenv

## 📦 Setup

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/pdf-chatbot.git
   cd pdf-chatbot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
  
3. **Set up environment variables**:
   Create a .env file in the root directory and add your Groq API key
   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Run the app**:
   ```bash
   streamlit run chatbot-app.py
   ```
