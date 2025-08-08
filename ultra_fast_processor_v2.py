"""
🚀 ULTRA-FAST DOCUMENT PROCESSOR - 6 API KEYS
=============================================
Optimized for <15 second response times
"""

import os
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils import extract_text_from_file, chunk_text
import google.generativeai as genai
import requests
import tempfile
from urllib.parse import urlparse
import hashlib
import time
import logging

logger = logging.getLogger(__name__)

class UltraFastDocumentProcessor:
    def __init__(self, speed_tier="ultra"):
        """SPEED-OPTIMIZED processor with 6 API keys"""
        load_dotenv()
        
        self.speed_tier = speed_tier
        
        # Load ALL 6 API keys
        self.api_keys = [
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("GOOGLE_API_KEY_2"),
            os.getenv("GOOGLE_API_KEY_3"),
            os.getenv("GOOGLE_API_KEY_4"),
            os.getenv("GOOGLE_API_KEY_5"),
            os.getenv("GOOGLE_API_KEY_6")
        ]
        
        self.api_keys = [key for key in self.api_keys if key and key.strip()]
        self.current_key_index = 0
        
        if not self.api_keys:
            raise ValueError("No valid Google API keys found!")
            
        genai.configure(api_key=self.api_keys[0])
        
        # Use fastest model
        self.model_name = "gemini-2.5-flash-lite"
        self.llm = genai.GenerativeModel("gemini-2.5-flash-lite")
        print(f"🚀 SPEED MODE: Using {self.model_name} with {len(self.api_keys)} API keys")
        
        # Use fast embeddings for speed
        print("⚡ Loading FAST embeddings model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Faster model
        
        # Document storage
        self.document_chunks = []
        self.document_embeddings = None
        self.faiss_index = None
        self.document_cache = {}
        
        print("✅ ULTRA-FAST processor ready!")
    
    def rotate_api_key(self):
        """Rotate through 6 API keys"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        current_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=current_key)
        self.llm = genai.GenerativeModel(self.model_name)
        logger.info(f"🔄 Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def fast_chunking(self, text):
        """AGGRESSIVE chunking for maximum speed"""
        import re
        
        # Very large chunks for speed
        chunk_size = 3000
        overlap = 100
        
        chunks = []
        
        # Try structural splitting first
        parts = re.split(r'(?=Part\s+[IVX]+)', text, flags=re.IGNORECASE)
        
        if len(parts) > 3:  # Good structure
            for part in parts:
                if len(part.strip()) < 200:
                    continue
                if len(part) <= chunk_size:
                    chunks.append(part.strip())
                else:
                    # Split into fixed-size chunks
                    for i in range(0, len(part), chunk_size - overlap):
                        chunk = part[i:i + chunk_size]
                        if len(chunk.strip()) > 200:
                            chunks.append(chunk.strip())
        else:
            # Fallback: simple fixed-size chunking
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if len(chunk.strip()) > 200:
                    chunks.append(chunk.strip())
        
        logger.info(f"⚡ FAST chunking: {len(chunks)} large chunks created")
        return chunks[:50]  # Limit chunks for speed
    
    def download_and_process_document(self, document_url):
        """Fast document download and processing"""
        try:
            url_hash = hashlib.md5(document_url.encode()).hexdigest()
            if url_hash in self.document_cache:
                logger.info("⚡ Using cached document")
                return self.document_cache[url_hash]
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(document_url, headers=headers, timeout=20, stream=True)
            response.raise_for_status()
            
            # Get file extension
            parsed_url = urlparse(document_url)
            file_extension = os.path.splitext(parsed_url.path)[1] or '.pdf'
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                temp_path = temp_file.name
            
            # Extract text
            document_text = extract_text_from_file(temp_path)
            os.unlink(temp_path)
            
            if not document_text or len(document_text.strip()) < 100:
                raise ValueError("Document is empty or too short")
            
            logger.info(f"⚡ Document processed: {len(document_text)} characters")
            self.document_cache[url_hash] = document_text
            return document_text
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {str(e)}")
            raise
    
    def process_document_fast(self, document_text):
        """SPEED-OPTIMIZED document processing"""
        try:
            logger.info("⚡ SPEED processing document...")
            
            # Fast chunking
            chunks = self.fast_chunking(document_text)
            
            if not chunks:
                raise ValueError("No chunks created")
            
            self.document_chunks = chunks
            
            # MAXIMUM SPEED embeddings
            logger.info("🚀 Generating ULTRA-FAST embeddings...")
            embeddings = self.embedder.encode(
                chunks, 
                batch_size=128,  # Maximum batch size
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Fast FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings)
            self.document_embeddings = embeddings
            
            logger.info(f"🚀 SPEED processing complete: {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {str(e)}")
            return False
    
    def fast_search(self, query, top_k=3):
        """ULTRA-FAST search"""
        try:
            if not self.document_chunks or self.faiss_index is None:
                return [], []
            
            query_embedding = self.embedder.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Minimal search for speed
            search_k = min(top_k * 2, len(self.document_chunks), 6)
            similarities, indices = self.faiss_index.search(query_embedding, search_k)
            
            results = []
            scores = []
            
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.document_chunks) and similarity > 0.4:  # High threshold
                    results.append(self.document_chunks[idx])
                    scores.append(float(similarity))
            
            return results[:top_k], scores[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Search failed: {str(e)}")
            return [], []
    
    def generate_fast_answer(self, question, relevant_chunks):
        """ULTRA-FAST answer generation"""
        try:
            if not relevant_chunks:
                return "No relevant information found."
            
            # Minimal context for speed
            context = "\n".join(relevant_chunks[:2])  # Only use top 2 chunks
            
            prompt = f"""Answer this question using ONLY the provided context. Be concise.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

            # Maximum speed generation
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=300,  # Short answers
                temperature=0.0,  # No randomness
                top_p=0.8,
                top_k=10,  # Minimal computation
                candidate_count=1
            )
            
            response = self.llm.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={"timeout": 10}  # Fast timeout
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                self.rotate_api_key()
                return "Unable to generate answer. Please try again."
                
        except Exception as e:
            if 'quota' in str(e).lower():
                self.rotate_api_key()
                time.sleep(0.5)
            return "Error processing question."
    
    def process_questions_fast(self, questions, document):
        """ULTRA-FAST multi-question processing"""
        try:
            logger.info(f"🚀 SPEED processing {len(questions)} questions...")
            
            # Clear cache for new document
            current_doc_hash = hashlib.md5(document.encode()).hexdigest()
            if not hasattr(self, '_current_doc_hash') or self._current_doc_hash != current_doc_hash:
                self.document_chunks = []
                self.faiss_index = None
                self._current_doc_hash = current_doc_hash
            
            # Process document if needed
            if not self.document_chunks:
                document_text = self.download_and_process_document(document)
                if not self.process_document_fast(document_text):
                    return ["Failed to process document."] * len(questions)
            
            answers = []
            
            for i, question in enumerate(questions, 1):
                logger.info(f"⚡ Question {i}/{len(questions)}")
                
                try:
                    # Fast search and answer
                    relevant_chunks, scores = self.fast_search(question, top_k=3)
                    answer = self.generate_fast_answer(question, relevant_chunks)
                    answers.append(answer)
                    
                except Exception as e:
                    logger.error(f"❌ Question {i} failed: {str(e)}")
                    answers.append("Error processing this question.")
            
            logger.info(f"🎉 SPEED processing complete!")
            return answers
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {str(e)}")
            return [f"Error: {str(e)}"] * len(questions)
