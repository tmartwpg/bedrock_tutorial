"""
LEVEL 8: Vector Store & RAG (Retrieval-Augmented Generation)

Building on Level 7, we now implement:
1. Document embedding and vectorization
2. Vector stores for similarity search
3. RAG (retrieval-augmented generation) patterns
4. Semantic search over documents
5. Grounding model responses with documents

Topics covered:
  - Embeddings
  - Vector stores (FAISS)
  - Document chunking
  - Semantic search
  - RAG workflows
  - Grounding with sources

Expected output:
  Model answers questions grounded in specific documents
"""

from typing import List
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from utils import pretty_print
from config import MODEL_ID, SAMPLE_DOCUMENT, VECTOR_STORE_SIMILARITY_TOP_K


class RAGTutorial:
    """
    Demonstrates RAG (Retrieval-Augmented Generation).

    RAG improves model accuracy by grounding answers in specific documents.
    """

    def __init__(self):
        """Initialize embeddings and model."""
        # Embeddings convert text to vectors
        self.embeddings = BedrockEmbeddings(
            region_name="us-east-1",
            model_id="amazon.titan-embed-text-v1"
        )

        # Chat model for generating responses
        self.model = ChatBedrock(
            model_id=MODEL_ID,
            region_name="us-east-1",
            model_kwargs={"temperature": 0.7}
        )

        self.vector_store = None

    def _create_sample_documents(self) -> List[Document]:
        """Create sample documents for the vector store."""
        docs = [
            Document(
                page_content=SAMPLE_DOCUMENT,
                metadata={"source": "bedrock_guide", "type": "overview"}
            ),
            Document(
                page_content="""Machine Learning is a subset of AI that focuses on learning from data.
                Models improve through experience without explicit programming.
                Common types include supervised learning, unsupervised learning, and reinforcement learning.""",
                metadata={"source": "ml_guide", "type": "definition"}
            ),
            Document(
                page_content="""Vector databases store embeddings - numerical representations of text.
                They enable semantic search by finding similar vectors.
                Common options include FAISS, Pinecone, Weaviate, and Milvus.
                FAISS is open-source and efficient for local use.""",
                metadata={"source": "vector_db_guide", "type": "technical"}
            ),
            Document(
                page_content="""LangChain is a framework for building applications with language models.
                It provides abstractions for common patterns like chains, agents, and memory.
                LangChain integrates with many LLM providers and vector databases.
                It simplifies building complex LLM applications.""",
                metadata={"source": "langchain_guide", "type": "framework"}
            ),
        ]
        return docs

    def example_1_embeddings_basics(self):
        """Example 1: Understanding embeddings."""
        pretty_print("Example 1: Embeddings Basics", "")

        texts = [
            "The cat sat on the mat",
            "A feline rested on the rug",
            "Python is a programming language",
        ]

        print("Creating embeddings for sample texts...")
        embeddings = self.embeddings.embed_documents(texts)

        print(f"\nGenerated {len(embeddings)} embeddings")
        print(f"Each embedding has {len(embeddings[0])} dimensions")
        print(f"First embedding (first 10 values): {embeddings[0][:10]}\n")

        # Show that similar texts have similar embeddings
        print("Similarity observations:")
        print("  Texts 1 & 2 are semantically similar (about sitting)")
        print("  Text 3 is about programming (different domain)")
        print("  Their embeddings reflect these relationships\n")

        print("[OK] Embeddings convert text to vectors!\n")

    def example_2_vector_store_creation(self):
        """Example 2: Creating and using a vector store."""
        pretty_print("Example 2: Vector Store Creation", "")

        # Get sample documents
        documents = self._create_sample_documents()
        print(f"Creating vector store with {len(documents)} documents...\n")

        # Split documents into chunks (for larger docs)
        splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separator="\n"
        )

        chunks = []
        for doc in documents:
            split_docs = splitter.split_documents([doc])
            chunks.extend(split_docs)

        print(f"Split into {len(chunks)} chunks")

        # Create vector store from chunks
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)

        print("[OK] Vector store created!\n")

    def example_3_semantic_search(self):
        """Example 3: Semantic search in vector store."""
        pretty_print("Example 3: Semantic Search", "")

        if not self.vector_store:
            self.example_2_vector_store_creation()

        # Search queries
        queries = [
            "What is machine learning?",
            "Tell me about AWS Bedrock",
            "What are vector databases?",
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            print("Most similar documents:")

            # Search for similar documents
            results = self.vector_store.similarity_search(
                query,
                k=VECTOR_STORE_SIMILARITY_TOP_K
            )

            for i, result in enumerate(results, 1):
                # Truncate content for display
                content = result.page_content[:150].replace("\n", " ")
                source = result.metadata.get("source", "unknown")
                print(f"  {i}. [{source}] {content}...")

        print("\n[OK] Semantic search finds relevant documents!\n")

    def example_4_rag_qa(self):
        """Example 4: RAG-based question answering."""
        pretty_print("Example 4: RAG Question Answering", "")

        if not self.vector_store:
            self.example_2_vector_store_creation()

        # Question to answer
        question = "What is AWS Bedrock and what can it do?"

        print(f"Question: {question}\n")

        # Step 1: Retrieve relevant documents
        print("Step 1: Retrieving relevant documents...")
        relevant_docs = self.vector_store.similarity_search(
            question,
            k=VECTOR_STORE_SIMILARITY_TOP_K
        )

        print(f"Found {len(relevant_docs)} relevant documents\n")

        # Step 2: Build context from documents
        context = "\n\n".join([
            f"[{doc.metadata.get('source', 'source')}]\n{doc.page_content}"
            for doc in relevant_docs
        ])

        # Step 3: Create RAG prompt
        rag_prompt = f"""Based on the following documents, answer the question.

DOCUMENTS:
{context}

QUESTION: {question}

Provide a clear answer based on the documents above."""

        print("Step 2: Generating answer from retrieved documents...")
        messages = [HumanMessage(content=rag_prompt)]
        response = self.model.invoke(messages)

        print(f"\nAnswer:\n{response.content}\n")
        print("[OK] RAG combines retrieval with generation!\n")

    def example_5_citations(self):
        """Example 5: Generate answers with citations."""
        pretty_print("Example 5: Answers with Citations", "")

        if not self.vector_store:
            self.example_2_vector_store_creation()

        question = "How does LangChain work?"

        # Retrieve documents
        relevant_docs = self.vector_store.similarity_search(question, k=2)

        # Build context with numbered sources
        context = ""
        for i, doc in enumerate(relevant_docs, 1):
            context += f"[Source {i}: {doc.metadata.get('source', 'unknown')}]\n"
            context += f"{doc.page_content}\n\n"

        # Create prompt asking for citations
        citation_prompt = f"""Answer this question using the sources below.
After each statement, include the source number in brackets like [Source 1].

SOURCES:
{context}

QUESTION: {question}

Answer with citations:"""

        messages = [HumanMessage(content=citation_prompt)]
        response = self.model.invoke(messages)

        print(f"Answer with citations:\n{response.content}\n")
        print("[OK] Sources can be cited for transparency!\n")

    def example_6_comparison_with_without_rag(self):
        """Example 6: Compare RAG vs non-RAG answers."""
        pretty_print("Example 6: RAG vs Non-RAG Comparison", "")

        question = "What are the key features of AWS Bedrock?"

        # Without RAG - just ask the model
        print("WITHOUT RAG (pure model knowledge):")
        print(f"Q: {question}")
        messages = [HumanMessage(content=question)]
        response_without = self.model.invoke(messages)
        print(f"A: {response_without.content[:300]}...\n")

        # With RAG - retrieve first
        print("WITH RAG (grounded in documents):")
        print(f"Q: {question}")

        if not self.vector_store:
            self.example_2_vector_store_creation()

        relevant_docs = self.vector_store.similarity_search(question, k=2)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        rag_prompt = f"""Based on these documents:
{context}

Answer: {question}"""

        messages = [HumanMessage(content=rag_prompt)]
        response_with = self.model.invoke(messages)
        print(f"A: {response_with.content[:300]}...\n")

        print("Differences:")
        print("  • Non-RAG: Uses training data, may be outdated or generic")
        print("  • RAG: Uses provided documents, more specific and current")
        print("  • RAG allows grounding in specific knowledge bases\n")
        print("[OK] RAG provides more controlled, accurate answers!\n")

    def demonstrate(self):
        """Run all examples."""
        print("\n" + "=" * 80)
        print("  LEVEL 8: Vector Store & RAG")
        print("=" * 80)

        try:
            self.example_1_embeddings_basics()
            self.example_2_vector_store_creation()
            self.example_3_semantic_search()
            self.example_4_rag_qa()
            self.example_5_citations()
            self.example_6_comparison_with_without_rag()

            # Summary
            pretty_print("Summary", "")
            print("Key takeaways:")
            print("  1. Embeddings convert text to vectors")
            print("  2. Vector stores enable semantic search")
            print("  3. Document chunking handles large texts")
            print("  4. RAG retrieves relevant context for generation")
            print("  5. Grounding improves accuracy and recency")
            print("  6. Citations provide transparency")
            print("  7. RAG is essential for domain-specific applications")

        except Exception as e:
            pretty_print("ERROR", str(e))
            raise


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """
    Entry point - demonstrates RAG patterns.

    Usage:
        python level_08_vector_store_rag.py
    """

    tutorial = RAGTutorial()
    tutorial.demonstrate()
