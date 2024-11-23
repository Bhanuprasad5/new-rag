import streamlit as st
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI


GOOGLE_API_KEY = "your-google-api-key"  # Replace with your API key
CHROMA_DB_DIR = "./chroma_db_"  # Directory for ChromaDB



# Initialize Chroma Database
db = Chroma(collection_name="vector_database",
            embedding_function=None,
            persist_directory=CHROMA_DB_DIR)

# Initialize Google Generative AI
genai_model = GoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-1.5-flash")

# Streamlit App
st.title("Question Answering with ChromaDB and Google GenAI")
st.write("Ask a question based on the context stored in the database.")

# Input Query
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Retrieving context and generating an answer..."):
        # Retrieve Context from ChromaDB
        docs_chroma = db.similarity_search_with_score(query, k=4)
        context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

        # Generate Answer
        PROMPT_TEMPLATE = """
        Answer the question based only on the following context:
        {context}
        Answer the question based on the above context: {question}.
        Provide a detailed answer.
        Don’t justify your answers.
        Don’t give information not mentioned in the CONTEXT INFORMATION.
        Do not say "according to the context" or "mentioned in the context" or similar.
        """
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query)

        response_text = genai_model.invoke(prompt)

    # Display Answer
    st.subheader("Answer:")
    st.write(response_text)
