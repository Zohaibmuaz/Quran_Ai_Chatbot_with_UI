# 📖 Quranic Insight AI

An advanced, multilingual RAG-based Chatbot for exploring the teachings of the Holy Quran, built with LangChain, OpenAI, and Streamlit. This project is designed to provide accurate, context-aware, and respectful answers grounded in authentic translations.

---

## 🌟 About The Project

This project was born from a simple but profound idea: "What if we could build a specialized AI to explore the Quran respectfully and accurately?" In an age of powerful Large Language Models (LLMs), the risk of "hallucination" (the AI making things up) is significant. For a text as sacred as the Holy Quran, this risk is unacceptable.

This RAG (Retrieval-Augmented Generation) system was built with a core principle: **zero tolerance for hallucination**. Every answer is grounded in and directly traceable to authentic translations of the Quranic text.

The journey was challenging, highlighting a surprising scarcity of clean, analysis-ready Islamic data. This project involved not just coding but also extensive data sourcing, cleaning, and iterative prompt engineering to create a system that is not only intelligent but also responsible. It stands as a testament to the idea that the true work of an engineer is to build the best possible solution within given constraints, especially when the subject matter demands the utmost care and respect.

This was a dream project, and its completion marks a significant milestone in applying modern AI to sacred knowledge in a safe and meaningful way.

### ✨ Key Features

- **Multilingual Interface:** Interacts seamlessly in both **English** and **Roman Urdu**.
- **Grounded Responses:** Answers are generated based *only* on a provided context of authentic translations, preventing the AI from fabricating information.
- **Structured Answers:** Provides clear, well-formatted responses that include an introductory summary, detailed points with translations, references, and explanations.
- **Advanced Retrieval:** Uses a powerful multilingual embedding model (`paraphrase-multilingual-mpnet-base-v2`) and a vector database (ChromaDB) to find relevant verses for a user's query.
- **Intelligent Prompting:** Employs a sophisticated "Chain of Thought" prompt that guides the LLM to identify the query language, select the appropriate translation, and format the output correctly.
- **Professional UI:** A beautiful, modern, and respectful user interface built with Streamlit.

### 🛠️ Built With

This project was made possible by these incredible technologies:

- **Python**
- **Streamlit** (for the web interface)
- **LangChain** (as the core RAG framework)
- **OpenAI GPT-4o** (as the Large Language Model)
- **ChromaDB** (as the vector store)
- **Sentence Transformers** (for embeddings)
- **Pandas** (for data manipulation)

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

You need to have Python 3.8+ installed on your system.

### Installation

1.  **Clone the repo**
    ```sh
    git clone [(https://github.com/Zohaibmuaz/Quran_Ai_Chatbot_with_UI.git]
    cd Quran_Ai_Chatbot_with_UI
    ```

2.  **Create a virtual environment (recommended)**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages**
    Create a `requirements.txt` file with the following content:
    ```txt
    pandas
    langchain
    langchain-community
    langchain-core
    langchain-openai
    streamlit
    python-dotenv
    chromadb
    sentence-transformers
    ```
    Then run:
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set up Data Files**
    Make sure the following four text files are present in your project's root directory:
    - `quran-simple.txt` (Arabic text)
    - `ur.maududi.txt` (Maududi's Urdu translation)
    - `ur.qadri.txt` (Tahir-ul-Qadri's Urdu translation)
    - `en_yusufali.txt` (Yusuf Ali's English translation)

5.  **Set up Environment Variables**
    Create a file named `.env` in the root directory. Add your OpenAI API key to this file:
    ```
    OPENAI_API_KEY="sk-..."
    ```

### Usage

The application runs in two steps:

1.  **First, prepare the master data file.** This script reads all the source `.txt` files and merges them into a single, structured `.csv` file. You only need to run this once.
    ```sh
    python prepare_data.py
    ```
    *(You will need to create this `prepare_data.py` file using the code we developed previously).*

2.  **Then, run the Streamlit application.**
    ```sh
    streamlit run app.py
    ```
    This will launch the web application in your browser. The first time you run this, it will take a few minutes to create the vector database. Subsequent launches will be much faster.

---

## 🚧 Known Limitations

While the system is powerful, it's important to understand its current limitations, which are common in RAG systems:

* **Narrative Retrieval:** The retriever struggles with broad, story-based queries like "Tell me the story of Adam and Iblees." It is better at finding verses related to specific concepts ("humility," "creation") than at piecing together a sequence of events.
* **Specificity:** Sometimes, for very specific theological terms (like `shirk`), the retriever might fail to find relevant verses if the query language (e.g., English) doesn't have a strong keyword overlap with the source text.

## 📈 Future Improvements

The current architecture is a strong foundation. Future work to overcome the limitations could involve implementing more advanced RAG techniques:

* **Metadata Filtering:** For queries like "Surah 2, Ayah 155," directly filtering the database by the `reference` metadata instead of relying on semantic search.
* **Advanced Retrieval Strategies:** Implementing techniques like HyDE (Hypothetical Document Embeddings) or Multi-Query Retriever to improve performance on narrative and complex questions.
* **Agentic RAG:** Building a master "Agent" that intelligently decides which retrieval tool (semantic search, metadata filter, etc.) to use based on the user's query.

---

## 📜 License

Distributed under the MIT License. See `LICENSE.txt` for more information.

---

## 🙏 Acknowledgments

* **Data Sources:** [Tanzil.net](https://tanzil.net) for the foundational Quranic text.
* **Core Frameworks:** The incredible open-source work by the teams behind [LangChain](https://www.langchain.com/), [Streamlit](https://streamlit.io/), and [ChromaDB](https://www.trychroma.com/).
