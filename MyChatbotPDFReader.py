import streamlit as st
import numpy as np
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory  # ✅ Correct import
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI

load_dotenv()


# Your existing functions (fixed get_pdf_text bug)
def get_pdf_text(pdf_data):
    text = ""
    for pdf in pdf_data:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text  # ✅ Fixed: return outside loop


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Your existing create_embeddings_vectorstore (unchanged)
def create_embeddings_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)

    # Your verification UI (unchanged)
    st.header("🔍 FAISS Verification")
    st.info(f"✅ **{len(vectorstore.index_to_docstore_id)}** vectors successfully embedded and stored")

    with st.expander("📊 Full Index Details"):
        st.json({"Total Vectors": len(vectorstore.index_to_docstore_id), "FAISS ntotal": vectorstore.index.ntotal})

    with st.expander("📝 All Stored Chunks"):
        all_docs = vectorstore.similarity_search("anything", k=len(vectorstore.index_to_docstore_id))
        for i, doc in enumerate(all_docs):
            st.write(f"**{i + 1}:** {doc.page_content}")

    st.subheader("🧪 Test Search")
    query = st.text_input("Enter test query:", "embedding")
    if st.button("Search") and query:
        results = vectorstore.similarity_search_with_score(query, k=5)
        st.dataframe([{"Content": doc.page_content[:150] + "...", "Score": f"{score:.4f}"}
                      for doc, score in results], use_container_width=True)

    return vectorstore, embeddings


# ✅ Fixed conversation chain
def get_conversation_chain(vectorstore, embeddings):
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.1,
        max_tokens=1000
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    #return conversation, memory
    return conversation_chain, None

def main():
    st.title("🤖 RAG Chatbot with Mistral")
    st.markdown("Upload PDFs and chat with your data!")

    if st.button("❌ Exit Chatbot"):
        st.warning("Closing the chatbot...")
        st.stop()


    # Session state for conversation
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.session_state.embeddings = None
        st.session_state.messages = []

    # Sidebar: Document upload
    with st.sidebar:
        st.subheader("📄 Your Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs", type="pdf", accept_multiple_files=True)

        if uploaded_files and st.button("🔄 Process Documents"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(uploaded_files)
                text_chunks = get_text_chunks(raw_text)

                vectorstore, embeddings = create_embeddings_vectorstore(text_chunks)
                st.session_state.conversation, _ = get_conversation_chain(vectorstore, embeddings)
                st.session_state.embeddings = embeddings
                st.success("✅ RAG Chain Ready!")

    # Main chat area
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input & response
    if prompt := st.chat_input("Ask about your PDFs..."):
        if st.session_state.conversation is None:
            st.error("❌ Please process documents first!")
            st.stop()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Mistral is answering..."):
                result = st.session_state.conversation({"question": prompt})
                response = result["answer"]
                st.markdown(response)

                # Show sources
                with st.expander(f"📚 Sources ({len(result.get('source_documents', []))})"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.write(f"**Source {i + 1}:** {doc.page_content[:300]}...")

            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()