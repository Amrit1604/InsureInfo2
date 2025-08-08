"""
🎯 UNIVERSAL DOCUMENT PROCESSOR - FREE & FAST
===========================================
Handles ANY document type with FREE Flash models:
- Insurance policies
- Constitution of India
- Vehicle manuals
- Medical documents
- Road safety guides
- Any PDF/DOCX document

FOCUS: SPEED + COST EFFICIENCY (FREE MODELS)
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

class UniversalDocumentProcessor:
    def __init__(self, high_quality_embeddings=True, speed_tier="ultra"):
        """Initialize with Flash models only - cost-effective and fast

        Args:
            high_quality_embeddings: Use higher quality embeddings
            speed_tier: "ultra" (2.5-flash-lite), "high" (2.0-flash-lite), "standard" (1.5-flash)
        """
        load_dotenv()

        # Model configuration - Flash models only for cost efficiency
        self.speed_tier = speed_tier

        # Load ALL 9 API keys for maximum capacity and failover
        self.api_keys = [
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("GOOGLE_API_KEY_2"),
            os.getenv("GOOGLE_API_KEY_3"),
            os.getenv("GOOGLE_API_KEY_4"),
            os.getenv("GOOGLE_API_KEY_5"),
            os.getenv("GOOGLE_API_KEY_6"),
            os.getenv("GOOGLE_API_KEY_7"),
            os.getenv("GOOGLE_API_KEY_8"),
            os.getenv("GOOGLE_API_KEY_9")  # 9th key for ultimate capacity
        ]

        # Filter valid keys
        self.api_keys = [key for key in self.api_keys if key and key.strip()]
        self.current_key_index = 0

        if not self.api_keys:
            raise ValueError("No valid Google API keys found!")

        # Initialize with first key and chosen Flash model
        genai.configure(api_key=self.api_keys[0])


        # Select the fastest available Flash model based on speed tier
        if speed_tier == "ultra":
            self.model_name = "gemini-2.0-flash-lite"
            self.llm = genai.GenerativeModel("gemini-2.0-flash-lite")
            print("🚀 Initializing with Gemini 2.0 Flash-Lite for ULTRA speed + FREE quota...")
        elif speed_tier == "high":
            self.model_name = "gemini-2.5-flash-lite"
            self.llm = genai.GenerativeModel("gemini-2.5-flash-lite")
            print("⚡ Initializing with Gemini 2.5 Flash-Lite for HIGH speed + FREE quota...")
        else:  # standard
            self.model_name = "gemini-1.5-flash"
            self.llm = genai.GenerativeModel("gemini-1.5-flash")
            print("⚡ Initializing with Gemini 1.5 Flash for standard speed + FREE quota...")

        # Embeddings model selection - EXTREME SPEED OPTIMIZATION
        print("⚡ Loading ULTRA-FAST embeddings model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Always use fastest model

        # Document storage
        self.document_chunks = []
        self.document_embeddings = None
        self.faiss_index = None
        self.document_metadata = []

        # Document cache
        self.document_cache = {}

        print("✅ Universal Document Processor initialized with FREE Flash models!")

    def rotate_api_key(self):
        """🔄 Smart rotation with failover - if one fails, try next key"""
        max_attempts = len(self.api_keys)  # Try all keys if needed

        for attempt in range(max_attempts):
            try:
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                current_key = self.api_keys[self.current_key_index]

                if not current_key:
                    logger.warning(f"⚠️ API key {self.current_key_index + 1} is empty, skipping...")
                    continue

                genai.configure(api_key=current_key)

                # Recreate Flash model with new key
                self.llm = genai.GenerativeModel(self.model_name)

                # Test the key with a simple call
                test_response = self.llm.generate_content(
                    "Hi",
                    generation_config=genai.types.GenerationConfig(max_output_tokens=5)
                )

                logger.info(f"✅ Successfully rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
                return True

            except Exception as e:
                logger.warning(f"⚠️ API key {self.current_key_index + 1} failed: {str(e)[:50]}")
                continue

        logger.error("❌ All API keys failed! Check your keys in .env file")
        return False

    def smart_generate_with_failover(self, prompt, generation_config=None):
        """🛡️ Generate with automatic failover to next API key if current fails"""
        max_key_attempts = min(3, len(self.api_keys))  # Try up to 3 keys

        for attempt in range(max_key_attempts):
            try:
                response = self.llm.generate_content(prompt, generation_config=generation_config)
                if response and response.text:
                    return response
                else:
                    raise Exception("Empty response")

            except Exception as e:
                logger.warning(f"⚠️ API key {self.current_key_index + 1} failed: {str(e)[:50]}")

                if attempt < max_key_attempts - 1:  # Don't rotate on last attempt
                    logger.info("🔄 Trying next API key...")
                    if not self.rotate_api_key():
                        break
                else:
                    logger.error("❌ All attempted API keys failed")
                    raise e

        raise Exception("All API keys exhausted")

    def create_intelligent_chunks(self, document_text):
        """Create ULTRA-FAST chunks while maintaining semantic accuracy"""
        import re

        logger.info("🚀 Applying ULTRA-FAST intelligent chunking...")

        # SPEED OPTIMIZATION: Simplified but accurate chunking strategy
        chunks = []

        # Strategy 1: Try structural splitting first (fast path)
        structural_patterns = [
            r'(?=Article\s+\d+)',  # Articles
            r'(?=Chapter\s+[IVX]+)',  # Chapters
            r'(?=Part\s+[IVX]+)',  # Parts
            r'(?=Section\s+\d+)',  # Sections
        ]

        # Fast structural split
        split_text = document_text
        for pattern in structural_patterns:
            parts = re.split(pattern, split_text, flags=re.IGNORECASE)
            if len(parts) > 3:  # Good structural division found
                split_text = parts
                break

        if isinstance(split_text, str):
            # No good structure found, use paragraph-based chunking (fast fallback)
            split_text = document_text.split('\n\n')

        # Strategy 2: Fast chunk assembly with optimal size (1800-2200 chars for accuracy)
        current_chunk = ""
        optimal_size = 2000

        for segment in split_text:
            segment = segment.strip()
            if not segment or len(segment) < 50:
                continue

            # Fast size check and assembly
            if len(current_chunk + segment) <= optimal_size:
                current_chunk += segment + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = segment + "\n\n"

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Strategy 3: Quick post-processing for accuracy
        final_chunks = []
        for chunk in chunks:
            if len(chunk) < 200:  # Too small, skip or merge
                if final_chunks and len(final_chunks[-1] + chunk) <= optimal_size:
                    final_chunks[-1] += "\n\n" + chunk
                continue
            final_chunks.append(chunk)

        logger.info(f"⚡ ULTRA-FAST chunking complete: {len(final_chunks)} chunks")
        logger.info(f"📊 Average chunk size: {sum(len(c) for c in final_chunks) // len(final_chunks)} chars")

        return final_chunks

    def download_and_process_document(self, document_url):
        """Download and process any document type with maximum accuracy"""
        try:
            logger.info(f"📥 Downloading document: {document_url[:100]}...")

            # Check cache
            url_hash = hashlib.md5(document_url.encode()).hexdigest()
            if url_hash in self.document_cache:
                logger.info("⚡ Using cached document")
                return self.document_cache[url_hash]

            # Download with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': '*/*',
                'Connection': 'keep-alive'
            }

            response = requests.get(document_url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()

            # Get file extension
            parsed_url = urlparse(document_url)
            file_extension = os.path.splitext(parsed_url.path)[1]

            if not file_extension:
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    file_extension = '.pdf'
                elif 'word' in content_type or 'docx' in content_type:
                    file_extension = '.docx'
                else:
                    file_extension = '.pdf'  # Default

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                temp_path = temp_file.name

            # Extract text with maximum accuracy
            logger.info(f"📄 Extracting text from {file_extension} document...")
            document_text = extract_text_from_file(temp_path)

            # Clean up
            os.unlink(temp_path)

            if not document_text or len(document_text.strip()) < 100:
                raise ValueError("Document is empty or too short")

            logger.info(f"✅ Document processed: {len(document_text)} characters extracted")

            # Cache the result
            self.document_cache[url_hash] = document_text

            return document_text

        except Exception as e:
            logger.error(f"❌ Document processing failed: {str(e)}")
            raise

    def process_document_with_accuracy(self, document_text):
        """🚀 EXTREME SPEED document processing - under 10 seconds target"""
        try:
            logger.info("🚀 EXTREME SPEED document processing...")

            # EXTREME OPTIMIZATION: Fewer, larger chunks for maximum speed
            chunks = self.create_speed_optimized_chunks(document_text)

            if not chunks:
                raise ValueError("No chunks created from document")

            logger.info(f"⚡ Created {len(chunks)} SPEED-OPTIMIZED chunks (target: 30-50 chunks max)")

            # Store chunks with minimal metadata
            self.document_chunks = chunks
            self.document_metadata = []

            for i, chunk in enumerate(chunks):
                self.document_metadata.append({
                    'chunk_id': i,
                    'chunk_text': chunk,
                    'char_count': len(chunk)
                })

            # EXTREME SPEED embeddings generation
            logger.info("⚡ EXTREME SPEED embeddings generation...")

            # Process ALL chunks at once with maximum batch size
            embeddings = self.embedder.encode(
                chunks,
                batch_size=len(chunks),  # Process everything at once
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device='cpu'
            )

            # SPEED-OPTIMIZED FAISS index creation
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Fastest index type
            self.faiss_index.add(embeddings)
            self.document_embeddings = embeddings

            logger.info(f"🚀 EXTREME SPEED processing complete: {len(chunks)} chunks in FAISS")
            return True

        except Exception as e:
            logger.error(f"❌ EXTREME SPEED processing failed: {str(e)}")
            return False

    def create_speed_optimized_chunks(self, text):
        """🚀 EXTREME SPEED: Minimal chunks for maximum speed"""
        try:
            # EXTREME OPTIMIZATION: Very large chunks, minimal count
            target_chunk_size = 8000  # Much larger chunks
            overlap = 100  # Minimal overlap

            chunks = []
            text_length = len(text)
            start = 0

            while start < text_length and len(chunks) < 20:  # Max 20 chunks total
                end = start + target_chunk_size

                # Simple break on paragraph or sentence
                if end < text_length:
                    break_point = text.rfind('\n\n', start, end)
                    if break_point == -1:
                        break_point = text.rfind('.', start, end)
                    if break_point != -1:
                        end = break_point + 1

                chunk = text[start:end].strip()
                if len(chunk) > 200:  # Only substantial chunks
                    chunks.append(chunk)

                start = max(end - overlap, start + 1)

            logger.info(f"⚡ EXTREME SPEED: {len(chunks)} large chunks (max 20)")
            return chunks

        except Exception as e:
            logger.error(f"❌ Speed chunking failed: {str(e)}")
            return []
            logger.info(f"⚡ Speed improvement: ~70% fewer chunks, ~60% faster embeddings")
            return True

        except Exception as e:
            logger.error(f"❌ Document processing failed: {str(e)}")
            return False

    def semantic_search_precise(self, query, top_k=8):
        """ULTRA-FAST semantic search with maintained accuracy"""
        try:
            if not self.document_chunks or self.faiss_index is None:
                logger.warning("No documents loaded for search")
                return [], []

            # SPEED OPTIMIZATION: Fast query embedding
            query_embedding = self.embedder.encode([query],
                                                 convert_to_numpy=True,
                                                 device='cpu')
            faiss.normalize_L2(query_embedding)

            # SPEED OPTIMIZATION: Reduced search scope but maintain accuracy
            search_k = min(top_k * 2, len(self.document_chunks), 20)  # Cap at 20 for speed
            similarities, indices = self.faiss_index.search(query_embedding, search_k)

            # Fast result assembly with accuracy threshold
            results = []
            scores = []

            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.document_chunks) and similarity > 0.15:  # Slightly lower threshold for speed
                    results.append(self.document_chunks[idx])
                    scores.append(float(similarity))

                if len(results) >= top_k:  # Fast exit when we have enough
                    break

            logger.info(f"⚡ ULTRA-FAST search: {len(results)} chunks found in minimal time")
            return results, scores

        except Exception as e:
            logger.error(f"❌ Semantic search failed: {str(e)}")
            return [], []

    def generate_accurate_answer(self, question, relevant_chunks, max_retries=3):
        """Generate highly accurate answer using document context"""
        try:
            if not relevant_chunks:
                logger.warning("No relevant chunks provided")
                return "No relevant information found in the document for this question."

            # Create comprehensive context
            context = "\n\n".join([f"CHUNK {i+1}:\n{chunk}" for i, chunk in enumerate(relevant_chunks)])

            # Ultra-detailed prompt for maximum accuracy
            prompt = f"""You are an expert document analyst. Your task is to provide the MOST ACCURATE answer based ONLY on the provided document content.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

CRITICAL INSTRUCTIONS:
1. Read the document context VERY CAREFULLY
2. Extract EXACT information from the document
3. Quote specific numbers, dates, percentages, and terms from the document
4. If the document contains multiple relevant sections, synthesize them accurately
5. If the exact answer is not in the document, clearly state "The document does not contain specific information about [topic]"
6. DO NOT make assumptions or add information not in the document
7. BE PRECISE with terminology used in the document
8. If there are conditions or exceptions mentioned, include them
9. Provide the most complete and accurate answer possible

ACCURATE ANSWER:"""

            # Try with retries and key rotation
            for attempt in range(max_retries):
                try:
                    logger.info(f"🎯 Generating accurate answer (attempt {attempt + 1})")

                    # SPEED OPTIMIZATION: Use failover system for reliable generation
                    generation_config = genai.types.GenerationConfig(
                        max_output_tokens=800,  # Slightly reduced for speed
                        temperature=0.1,  # Keep low for accuracy
                        top_p=0.9,  # Slightly higher for speed
                        top_k=40,
                        candidate_count=1
                    )

                    response = self.smart_generate_with_failover(prompt, generation_config)

                    if response and response.text:
                        answer = response.text.strip()
                        logger.info(f"✅ Accurate answer generated: {len(answer)} characters")
                        return answer
                    else:
                        logger.warning("Empty response, retrying...")

                except Exception as e:
                    error_msg = str(e).lower()
                    logger.error(f"❌ Answer generation failed: {str(e)}")

                    if 'quota' in error_msg or 'rate' in error_msg:
                        logger.info("🔄 Quota exceeded, rotating API key...")
                        self.rotate_api_key()
                        time.sleep(1)
                    elif 'timeout' in error_msg:
                        logger.warning("⏰ Timeout, retrying...")
                        time.sleep(2)
                    else:
                        break

            # Fallback response
            return "Unable to generate accurate answer due to technical limitations. Please try again."

        except Exception as e:
            logger.error(f"❌ Answer generation failed: {str(e)}")
            return "Error processing the question. Please contact support."

    def process_question_with_maximum_accuracy(self, question, document):
        """Process a single question with maximum possible accuracy"""
        try:
            logger.info(f"🎯 Processing question with maximum accuracy: {question[:100]}...")

            # Step 1: Ensure document is loaded and processed
            if not self.document_chunks:
                logger.info("📄 Loading and processing document...")
                document_text = self.download_and_process_document(document)
                if not self.process_document_with_accuracy(document_text):
                    return "Failed to process the document. Please check the document URL."

            # Step 2: Perform precise semantic search
            relevant_chunks, scores = self.semantic_search_precise(question, top_k=8)

            if not relevant_chunks:
                return "No relevant information found in the document for this question."

            # Step 3: Generate accurate answer
            answer = self.generate_accurate_answer(question, relevant_chunks)

            logger.info(f"✅ Question processed with maximum accuracy")
            return answer

        except Exception as e:
            logger.error(f"❌ Question processing failed: {str(e)}")
            return f"Error processing question: {str(e)}"

    def process_multiple_questions_accurately(self, questions, document):
        """🚀 ULTRA-FAST batch processing for multiple questions"""
        try:
            logger.info(f"🚀 SPEED-OPTIMIZED processing {len(questions)} questions...")

            # ALWAYS clear and reload for each new request to ensure dynamic processing
            current_doc_hash = hashlib.md5(document.encode()).hexdigest()

            # Check if this is a different document than last time
            if not hasattr(self, '_current_doc_hash') or self._current_doc_hash != current_doc_hash:
                logger.info("🔄 New document detected, clearing cached chunks and reprocessing...")
                self.document_chunks = []
                self.document_embeddings = None
                self.faiss_index = None
                self.document_metadata = []
                self._current_doc_hash = current_doc_hash

            # Load and process document if we don't have chunks
            if not self.document_chunks:
                logger.info("📄 Loading and processing document for all questions...")
                document_text = self.download_and_process_document(document)
                if not self.process_document_with_accuracy(document_text):
                    return ["Failed to process the document. Please check the document URL."] * len(questions)

            # 🚀 SPEED OPTIMIZATION: Batch process all questions together
            return self._batch_process_questions_ultra_fast(questions)

        except Exception as e:
            logger.error(f"❌ Batch processing failed: {str(e)}")
            return [f"Error processing questions: {str(e)}"] * len(questions)

    def _batch_process_questions_ultra_fast(self, questions):
        """🚀 EXTREME SPEED: Process all questions in 1-2 API calls"""
        try:
            logger.info("🚀 Starting EXTREME SPEED batch processing...")

            # Step 1: Batch embed all questions at once (much faster)
            logger.info("📊 Batch embedding all questions...")
            question_embeddings = self.embedder.encode(
                questions,
                batch_size=len(questions),  # Process all at once
                show_progress_bar=False,
                device='cpu',
                normalize_embeddings=True
            )

            # Step 2: Find top chunks across ALL questions (aggressive filtering)
            logger.info("🔍 Finding best chunks across all questions...")
            all_chunk_scores = {}

            for question_embedding in question_embeddings:
                scores, indices = self.faiss_index.search(
                    question_embedding.reshape(1, -1).astype('float32'),
                    k=10  # Get more candidates
                )

                # Accumulate scores for each chunk
                for idx, score in zip(indices[0], scores[0]):
                    if score > 0.25:  # Lower threshold for more content
                        if idx not in all_chunk_scores:
                            all_chunk_scores[idx] = 0
                        all_chunk_scores[idx] += score

            # Get top chunks globally (best across all questions)
            top_chunk_indices = sorted(all_chunk_scores.keys(),
                                     key=lambda x: all_chunk_scores[x],
                                     reverse=True)[:12]  # Top 12 chunks total

            top_chunks = [self.document_chunks[idx] for idx in top_chunk_indices]

            # Step 3: MEGA BATCH - Process ALL questions in one API call
            logger.info("💡 MEGA BATCH: Processing all questions in one call...")
            return self._generate_mega_batch_answers(questions, top_chunks)

        except Exception as e:
            logger.error(f"❌ Extreme speed processing failed: {str(e)}")
            return ["Error in extreme speed processing"] * len(questions)

    def _generate_mega_batch_answers(self, questions, chunks):
        """Generate ALL answers in one mega API call"""
        try:
            # Create comprehensive context from top chunks
            context = "\n\n".join(chunks[:8])  # Use top 8 chunks only

            # Create mega prompt for all questions
            questions_text = ""
            for i, question in enumerate(questions, 1):
                questions_text += f"\n{i}. {question}"

            prompt = f"""Based on the document context below, answer ALL questions concisely and accurately.

QUESTIONS:{questions_text}

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
- Answer each question directly and concisely
- Use only information from the provided context
- Keep each answer under 100 words
- Format as numbered list: 1. [answer] 2. [answer] etc.
- If context doesn't contain answer, say "Information not available in document"

ANSWERS:"""

            # Single API call for all questions with failover
            response = self.smart_generate_with_failover(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1200,  # Enough for all answers
                    temperature=0.05,  # Very focused
                    top_p=0.9,
                    top_k=30
                )
            )

            if response and response.text:
                # Parse mega response
                response_text = response.text.strip()

                # Extract numbered answers
                answers = []
                lines = response_text.split('\n')
                current_answer = ""

                for line in lines:
                    line = line.strip()
                    # Check if line starts with a number
                    if line and any(line.startswith(f"{i}.") for i in range(1, len(questions) + 1)):
                        if current_answer:
                            answers.append(current_answer.strip())
                        # Remove number prefix and start new answer
                        number_end = line.find('.')
                        if number_end != -1:
                            current_answer = line[number_end+1:].strip()
                        else:
                            current_answer = line
                    elif current_answer and line:
                        current_answer += " " + line

                # Add final answer
                if current_answer:
                    answers.append(current_answer.strip())

                # Ensure we have answers for all questions
                while len(answers) < len(questions):
                    answers.append("Information not available in document.")

                # Trim to exact number of questions
                return answers[:len(questions)]

            else:
                return ["Unable to generate response from the document."] * len(questions)

        except Exception as e:
            logger.error(f"❌ Mega batch generation failed: {str(e)}")
            return [f"Error: {str(e)[:50]}"] * len(questions)

    def _generate_batch_answers(self, batch_questions, batch_chunks):
        """Generate answers for a batch of questions with speed optimization"""
        try:
            # Combine all context for batch processing
            combined_context = ""
            for i, chunks in enumerate(batch_chunks):
                if chunks:
                    context = "\n".join(chunks[:3])  # Top 3 chunks only
                    combined_context += f"\n\nContext for Question {i+1}:\n{context}"

            # Create batch prompt
            questions_text = ""
            for i, question in enumerate(batch_questions):
                questions_text += f"\n{i+1}. {question}"

            prompt = f"""Based on the provided document context, answer these questions quickly and accurately:

QUESTIONS:{questions_text}

CONTEXT:{combined_context}

Provide direct, concise answers. Format as:
1. [Answer to question 1]
2. [Answer to question 2]
3. [Answer to question 3]

Keep answers focused and under 150 words each."""

            # Generate with speed settings
            self.rotate_api_key()  # Use API key rotation
            response = self.llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=600,  # Reduced for speed
                    temperature=0.1,  # Lower for consistency
                    top_p=0.8,  # Faster sampling
                    top_k=20   # Faster sampling
                )
            )

            # Parse batch response
            if response and response.text:
                # Split response into individual answers
                response_text = response.text.strip()

                # Extract numbered answers
                answers = []
                lines = response_text.split('\n')
                current_answer = ""

                for line in lines:
                    line = line.strip()
                    if line and any(line.startswith(f"{i}.") for i in range(1, len(batch_questions) + 1)):
                        if current_answer:
                            answers.append(current_answer.strip())
                        current_answer = line[2:].strip()  # Remove number prefix
                    elif current_answer:
                        current_answer += " " + line

                # Add last answer
                if current_answer:
                    answers.append(current_answer.strip())

                # Ensure we have answers for all questions
                while len(answers) < len(batch_questions):
                    answers.append("Unable to generate answer from the provided context.")

                return answers[:len(batch_questions)]

            else:
                return ["Unable to generate answer from the provided context."] * len(batch_questions)

        except Exception as e:
            logger.error(f"❌ Batch answer generation failed: {str(e)}")
            return [f"Error generating answer: {str(e)[:100]}"] * len(batch_questions)
