![An implementaion of RAG](rag.jpg)

# Chat with Local Documents

This project is a document-based conversational assistant that allows users to query local documents using **RAG (Retrieval-Augmented Generation)** and interact with an **LLM (Large Language Model)** to get intelligent responses. The application is built using **Streamlit** for the frontend and integrates tools like **Chroma**, **HuggingFace models**, and **SentenceTransformer embeddings**.

---

## Features

- **Retrieve Contextual Information**: Query local documents stored in a directory, and retrieve relevant sections based on user questions.
- **Conversational AI**: Use an AI assistant to provide responses based on the retrieved context.
- **Caching for Efficiency**: Cache document loading, chunking, and embeddings to improve performance.
- **Customizable Models**: Plug in different HuggingFace models as needed for text generation and embeddings.

---

## Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python, HuggingFace, Chroma, SentenceTransformer  
- **Environment Variables**: Managed using `dotenv`

---

## Setup and Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Mekazstan/langchain_rag_agent
   cd langchain_rag_agent
   ```

2. **Install Dependencies**  
   Ensure you have Python 3.8+ installed. Then, run:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**  
   Create a `.env` file in the project root and define the following variables:  
   ```env
   LLM_MODEL=meta-llama/Meta-Llama-3.1-405B-Instruct
   DIRECTORY=books
   ```

   Replace `meta-llama/Meta-Llama-3.1-405B-Instruct` and `books` with your model and document directory.

4. **Run the Application**  
   Start the Streamlit app:  
   ```bash
   streamlit run agent.py
   ```

---

## Usage

1. **Launch the App**  
   Open the app in your browser (default: `http://localhost:8501`).

2. **Interact with the AI**  
   - Type your questions in the input box.  
   - The AI retrieves relevant context from the documents and generates an intelligent response.  

3. **Example Questions**  
   - *"What are the key points in document X?"*  
   - *"Summarize the content of document Y."*  

---

## Project Structure

- **`agent.py`**: Main Streamlit application file.  
- **`requirements.txt`**: List of Python dependencies.  
- **`documents/`**: Folder to store your documents for RAG.  
- **`.env`**: Environment configuration file.  

---

## How It Works

1. **Load Documents**  
   Documents are loaded from a specified directory and split into smaller chunks using `CharacterTextSplitter`.

2. **Generate Embeddings**  
   Chunks are converted into vector embeddings using `SentenceTransformerEmbeddings`.

3. **Build a Vector Store**  
   The embeddings are stored in a **Chroma** vector database for fast similarity-based retrieval.

4. **Query and Generate Responses**  
   - User queries are processed to find similar document chunks.  
   - The retrieved context is passed to the AI model for generating responses.

---

## Customization

- **Model**: Update the `LLM_MODEL` environment variable to use a different HuggingFace model.  
- **Documents Directory**: Set the `DIRECTORY` environment variable to change the location of your documents.  
- **Embedding Model**: Modify `SentenceTransformerEmbeddings` to use another SentenceTransformer model.  

---

## Future Improvements

- Enhance user interface for better interaction (Using Frontend technologies).  
- Integrate support for multi-user chat sessions & user authentication.  

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
