import streamlit as st
import pandas as pd
import os
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Quranic Insight AI",
    page_icon="🕋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS for Theming and Design ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&family=Amiri&display=swap');

    /* Main background with geometric pattern */
    .stApp {
        background-color: #1a1a1a; /* Dark Charcoal */
        background-image: linear-gradient(315deg, rgba(255, 255, 255, 0.02) 25%, transparent 25%),
                          linear-gradient(45deg, rgba(255, 255, 255, 0.02) 25%, transparent 25%);
        background-size: 20px 20px;
        color: #e0e0e0; /* Off-white text */
    }

    /* Main title font and color */
    h1 {
        font-family: 'Amiri', serif;
        color: #d4af37; /* Soft Gold */
        text-align: center;
        padding-top: 2rem;
    }

    /* Subtitle style */
    .subtitle {
        font-family: 'Merriweather', serif;
        color: #b0b0b0;
        text-align: center;
        font-size: 1.1rem;
    }
    
    /* Sidebar styling */
    .st-emotion-cache-16txtl3 {
        background-color: #212121;
    }

    /* Chat message styling */
    .st-emotion-cache-1c7y2kd { /* Chat message container */
        background-color: rgba(42, 42, 42, 0.8);
        border: 1px solid #d4af37;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    /* Input box styling */
    .st-emotion-cache-1jicfl2 {
        background-color: #2a2a2a;
    }

    /* Output formatting improvements */
    .stMarkdown h3 {
        color: #50c878; /* Mint Green for headings */
        border-bottom: 2px solid #d4af37;
        padding-bottom: 5px;
    }
    .stMarkdown blockquote {
        background-color: rgba(212, 175, 55, 0.1);
        border-left: 5px solid #d4af37;
        padding: 0.5rem 1rem;
        margin-left: 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# --- 3. Cached Functions for Heavy Lifting ---
@st.cache_resource
def load_rag_chain():
    load_dotenv()
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")

    csv_filename = 'quran_multilingual_data.csv'
    if not os.path.exists(csv_filename):
        st.error(f"CRITICAL ERROR: The data file '{csv_filename}' was not found.")
        st.stop()
    
    df = pd.read_csv(csv_filename)
    df.fillna("", inplace=True)

    df['page_content'] = "Reference: " + df['reference'].astype(str) + "\n" + \
                         "Urdu Translation 1: " + df['translation_maududi'] + "\n" + \
                         "Urdu Translation 2: " + df['translation_qadri'] + "\n" + \
                         "English Translation: " + df['translation_english']
    
    loader = DataFrameLoader(df, page_content_column='page_content')
    documents = loader.load()

    persist_directory = "./quran_multilingual_db"
    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        with st.spinner("Creating new multilingual database. This might take a few minutes..."):
            vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)

    retriever = vectorstore.as_retriever(search_kwargs={'k': 7})
    
    # --- New and Improved Prompt Template for Better Formatting ---
    prompt_template = """
    You are an expert and respectful Quranic Assistant. Your task is to follow a strict, step-by-step process to answer the user's question based ONLY on the context, using precise Markdown formatting.

    **Your Thought Process (Follow these steps internally):**
1.  **Step 1: Identify Language.** Analyze the user's `Question` to determine if it is in English or Roman Urdu. This decision is critical and will control the language of your entire response.
2.  **Step 2: Synthesize a Summary.** Based on the language identified in Step 1, carefully read the user's question and understand it and then read the `Context` and formulate a 3-4 line summary that directly answers the `Question`.
3.  **Step 3: Format Detailed Points.** Create a numbered list of key points from the `Context`. For each point, you must follow these sub-rules precisely:
    -   **Sub-rule 3a:** If the identified language was English, you MUST use the "English Translation" from the context for the `Translation:` field.
    -   **Sub-rule 3b:** If the identified language was Roman Urdu, you MUST use one of the "Urdu Translation" texts from the context for the `Translation:` field.
    -   **Sub-rule 3c:** The `Explanation:` must be in the same language as the `Question`.

    ---

    ### Detailed Points
    (Create a numbered list of key points below.)

    1.  **Translation:**
        > (The appropriate translation text goes here, inside a blockquote.)
        **Reference:** `[The verse reference, e.g., 2:153]`
        
        **Explanation:** (Your 1-2 line explanation for this point goes here.)

    2.  **Translation:**
        > (The second translation text goes here.)
        **Reference:** `[The second verse reference]`
        
        **Explanation:** (The explanation for the second point.)
    
    (and so on...Try to give as much points as you can generate)

    **Context from Database:**
    {context}

    **User's Question:**
    {question}

    **Your Final Answer (Strictly follow the Markdown format above):**
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- 4. Main App Interface ---

# Load the RAG chain (fast due to caching)
rag_chain = load_rag_chain()

# Sidebar for information
with st.sidebar:
    st.title("About Quranic Insight AI")
    st.markdown("""
    This is an AI-powered assistant designed to help you explore the teachings of the Holy Quran. 
    
    **How it works:**
    1.  Ask a question in English or Roman Urdu.
    2.  The AI searches through multiple translations of the Quran to find the most relevant verses.
    3.  It then uses a powerful language model to generate a structured and informative answer based on those verses.
    
    **Data Sources:**
    - Arabic Text: Tanzil.net
    - Urdu Translations: Maududi & Tahir-ul-Qadri
    - English Translation: Abdullah Yusuf Ali
    """)
    st.info("This is an experimental AI project. Always consult with a qualified Islamic scholar for definitive religious guidance.")

# Main page title
st.title("Quranic Insight AI | قرآنی معاون")
st.markdown("<p class='subtitle'>Your AI assistant for exploring the Quran</p>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "As-salamu alaykum! How can I help you explore the Quran today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the Quran..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Analyzing verses..."):
            response = rag_chain.invoke(prompt)
            st.markdown(response, unsafe_allow_html=True)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})