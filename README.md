# News Research Tool 📈

The **News Research Tool** is a Streamlit-based web application designed to extract, process, and analyze news articles. Users can input URLs of news articles, and the application will create embeddings, build a vector store using FAISS, and enable retrieval-based question answering (QA) with sources.

---

## Features
- Extract content from news article URLs.
- Split content into manageable chunks for embedding.
- Generate embeddings using OpenAI models.
- Store and load embeddings with FAISS.
- Perform question answering with sources using LangChain.
- Display answers and reference sources interactively.

---

## Requirements
- Python 3.9 or higher
- OpenAI API key (for embeddings and LLM usage)

### **Python Libraries**
The following libraries are required:
- `streamlit`
- `langchain`
- `openai`
- `pickle`
- `faiss`
- `python-dotenv`

Install dependencies using:
```bash
pip install streamlit langchain openai faiss-cpu python-dotenv
```
---

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/news-research-tool.git
   cd news-research-tool
   ```

2. **Set Up Environment Variables**
   Create a `.env` file in the project directory and add your OpenAI API key:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Run the Application**
   Start the Streamlit app:
   ```bash
   streamlit run main.py
   ```

4. **Interact with the Tool**
   - Enter up to three news article URLs in the sidebar.
   - Click the **"Process URLs"** button to load and process the articles.
   - Enter a question in the text box to retrieve answers and sources.

---

## Application Workflow

1. **Load and Process URLs**
   - The `UnstructuredURLLoader` fetches article content from the provided URLs.
   - The content is split into chunks using the `RecursiveCharacterTextSplitter`.

2. **Generate Embeddings**
   - OpenAI embeddings are created using the `OpenAIEmbeddings` class.
   - FAISS stores the embeddings in a local vector index (`faiss_index`).

3. **Question Answering**
   - The FAISS index is used as a retriever via the `vectorstore_openai.as_retriever()` method.
   - The `RetrievalQAWithSourcesChain` uses an OpenAI LLM to generate answers and cite sources.

4. **Display Results**
   - The tool displays the answer and sources in an interactive Streamlit interface.

---

## Code Structure
### **Key Components**
- **Document Loader:** Extracts article content using `UnstructuredURLLoader`.
- **Text Splitter:** Splits the text into smaller, overlapping chunks for embeddings.
- **Embeddings:** Generates vector embeddings using OpenAI's models.
- **Vector Store:** Stores embeddings using FAISS for fast retrieval.
- **Retrieval Chain:** Uses LangChain's `RetrievalQAWithSourcesChain` for answering questions.

---

## Security Note
The tool relies on pickle-based deserialization for loading the FAISS index. This can pose security risks if the pickle file is tampered with. To mitigate this:
- Only use pickle files you trust.
- Deserialization is explicitly enabled with `allow_dangerous_deserialization=True`.

---

## Future Enhancements
- Support for additional languages.
- Integration with more advanced LLMs or embedding models.
- Automated summarization of articles.
- Enhanced UI/UX for better user interaction.

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the tool.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Enjoy researching your favorite news articles with the **News Research Tool**! 🎉
