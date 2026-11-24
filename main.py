import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

warnings.filterwarnings("ignore")

load_dotenv()

PDF_PATH = Path(__file__).resolve().parent / "Java_Previous_Year_Questions_Answers.pdf"
COLLECTION_NAME = "java_exam_collection"


def load_java_document():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Java PDF not found: {PDF_PATH}")

    print(f"Loading document from: {PDF_PATH}")
    documents = PyPDFLoader(str(PDF_PATH)).load()
    print(f"Loaded {len(documents)} document pages.")
    return documents


def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    print("Creating embeddings and indexing into Qdrant vector store...")
    embeddings = FastEmbedEmbeddings()
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_documents(chunks)
    print("Indexing complete!")
    return vector_store


def create_rag_chain(vector_store):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to the workspace .env file before running."
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.2,
        max_tokens=2500,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Java programming tutor. "
                "Answer accurately using ONLY the provided context from the Java exam PDF. "
                "Use clear explanations and include Java code when the question asks for a program. "
                "If the answer is not in the context, say that it is not covered by the provided PDF.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever, prompt | llm | StrOutputParser()


def query_rag(question: str, retriever, rag_chain):
    print("\n" + "=" * 60)
    print(f"Query: {question}")
    print("=" * 60)

    retrieved_docs = retriever.invoke(question)
    print(f"\nRetrieved {len(retrieved_docs)} relevant context chunks:")
    for index, doc in enumerate(retrieved_docs, start=1):
        page_number = doc.metadata.get("page", "Unknown")
        preview = doc.page_content.strip().replace("\n", " ")[:150]
        print(f" [{index}] Page {page_number}: {preview}...")

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    print("\nGenerating response...")
    response = rag_chain.invoke({"context": context, "question": question})
    print("\nAnswer:")
    print(response)
    print("=" * 60)
    return response


if __name__ == "__main__":
    documents = load_java_document()
    vector_store = create_vector_store(documents)
    retriever, rag_chain = create_rag_chain(vector_store)

    default_question = (
        "What are the differences between checked and unchecked exceptions in Java?"
    )
    query_rag(default_question, retriever, rag_chain)
