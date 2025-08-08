"""
🎯 STREAMLINED POLICY ANALYZER
==============================
Fast, focused analysis of specific policy documents
"""

import os
import requests
import tempfile
import time
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF processing
from typing import List, Tuple
import numpy as np
import faiss

class StreamlinedPolicyAnalyzer:
    def __init__(self):
        """Initialize with minimal setup"""
        load_dotenv()

        # Setup API key
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        self.llm = genai.GenerativeModel("gemini-1.5-flash")

        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Document cache
        self.document_text = None
        self.document_chunks = []
        self.embeddings = None
        self.index = None

    def download_and_extract_pdf(self, url: str) -> str:
        """Download and extract text from PDF URL"""
        try:
            print(f"📥 Downloading PDF from URL...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name

            # Extract text using PyMuPDF
            print(f"📄 Extracting text from PDF...")
            doc = fitz.open(temp_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # Clean up
            os.unlink(temp_path)

            print(f"✅ Extracted {len(text)} characters from PDF")
            return text

        except Exception as e:
            print(f"❌ PDF extraction failed: {str(e)}")
            return None

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def load_document(self, url: str) -> bool:
        """Load and process document"""
        self.document_text = self.download_and_extract_pdf(url)
        if not self.document_text:
            return False

        # Create chunks
        self.document_chunks = self.chunk_text(self.document_text)
        print(f"📚 Created {len(self.document_chunks)} chunks")

        # Generate embeddings
        print(f"🧠 Generating embeddings...")
        chunk_embeddings = self.embedding_model.encode(self.document_chunks)

        # Create FAISS index
        dimension = chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(chunk_embeddings.astype('float32'))

        print(f"✅ Document processing complete!")
        return True

    def search_document(self, query: str, top_k: int = 3) -> List[str]:
        """Search document for relevant chunks"""
        if not self.index:
            return []

        # Encode query
        query_embedding = self.embedding_model.encode([query])

        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Return relevant chunks
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            if scores[0][i] > 0.3:  # Similarity threshold
                relevant_chunks.append(self.document_chunks[idx])

        return relevant_chunks

    def analyze_question(self, question: str) -> str:
        """Analyze question using document content"""
        try:
            # Search for relevant content
            relevant_chunks = self.search_document(question, top_k=5)

            if not relevant_chunks:
                return "No relevant information found in the document. Please contact customer service."

            # Create context
            context = "\n\n".join(relevant_chunks)

            # Create prompt
            prompt = f"""Based on the specific policy document content below, answer this exact question:

QUESTION: {question}

POLICY DOCUMENT CONTENT:
{context}

INSTRUCTIONS:
- Extract the EXACT answer from the policy document
- Quote specific numbers, periods, percentages, and conditions mentioned
- If the document contains the answer, provide it precisely
- If multiple conditions apply, list them clearly
- Be specific and accurate based on the document content
- Do not add information not in the document

ANSWER:"""

            # Get LLM response
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=300,
                temperature=0.1,
                candidate_count=1
            )

            response = self.llm.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={"timeout": 10}
            )

            if response and response.text:
                return response.text.strip()
            else:
                return "Unable to process the question. Please contact customer service."

        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            return "Processing error occurred. Please contact customer service."

def test_analyzer():
    """Test the streamlined analyzer"""
    print("🚀 Testing Streamlined Policy Analyzer")
    print("=" * 50)

    analyzer = StreamlinedPolicyAnalyzer()

    # Test URL
    url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

    # Load document
    start_time = time.time()
    success = analyzer.load_document(url)
    load_time = time.time() - start_time

    if not success:
        print("❌ Document loading failed!")
        return

    print(f"✅ Document loaded in {load_time:.2f}s")

    # Test questions
    questions = [
        "What is the specified Grace Period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "How does the policy define 'Hospital' and what are the minimum facility requirements?",
        "What coverage does the policy provide for cataract surgery and under what conditions may limits not apply?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n🔍 Question {i}: {question}")
        start_time = time.time()
        answer = analyzer.analyze_question(question)
        analysis_time = time.time() - start_time
        print(f"⏱️ Analysis time: {analysis_time:.2f}s")
        print(f"📝 Answer: {answer}")
        print("-" * 50)

if __name__ == "__main__":
    test_analyzer()
