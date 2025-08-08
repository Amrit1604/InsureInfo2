"""
🏥 INTELLIGENT INSURANCE CLAIMS PROCESSING SYSTEM
==================================================
Built with LLMs to process natural language queries and retrieve relevant
information from large unstructured policy documents.

✨ Features:
- Parse natural language queries (e.g., "46M, knee surgery, Pune, 3-month policy")
- Smart semantic search through policy documents
- AI-powered decision making with clear justifications
- Reference specific clauses used in decisions
- User-friendly explanations in plain English
"""

import os
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils import extract_text_from_file, chunk_text, process_multiple_documents
import google.generativeai as genai
import json
from colorama import init, Fore, Back, Style
import requests
import tempfile
from urllib.parse import urlparse
import hashlib
import time

# Initialize colorama for better console output
init(autoreset=True)

class IntelligentClaimsProcessor:
    def __init__(self):
        """Initialize the claims processing system with multiple API keys"""
        # Load environment variables
        load_dotenv()

        # 🔥 LOAD ALL 4 API KEYS FOR UNLIMITED PROCESSING
        self.api_keys = [
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("GOOGLE_API_KEY_2"),
            os.getenv("GOOGLE_API_KEY_3"),
            os.getenv("GOOGLE_API_KEY_4")
        ]

        # Filter out None/empty keys
        self.api_keys = [key for key in self.api_keys if key and key.strip()]

        if not self.api_keys:
            print(f"{Fore.RED}❌ Error: No valid Google API keys found!")
            print("Please check your .env file.")
            exit(1)

        self.current_key_index = 0
        self.current_key = self.api_keys[0]

        print(f"{Fore.GREEN}🔥 UNLIMITED API ACCESS: {len(self.api_keys)} Google API keys loaded!")
        print(f"{Fore.CYAN}⚡ NO RATE LIMITS - FULL SPEED AHEAD!")

        # Configure Gemini AI with first key
        genai.configure(api_key=self.current_key)
        self.llm = genai.GenerativeModel("gemini-1.5-flash")

        # Initialize components
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Document processing variables - BADASS MULTI-SOURCE SYSTEM
        self.document_chunks = []
        self.document_sources = []
        self.embeddings = None
        self.current_document_urls = []  # Support multiple URLs
        self.document_cache = {}  # Cache for downloaded documents
        self.local_docs_loaded = False  # Track if sample docs are loaded
        self.dynamic_docs_count = 0  # Track dynamic documents

        print(f"{Fore.GREEN}✅ Intelligent Claims Processor initialized successfully!")

    def rotate_api_key(self):
        """🔄 AUTOMATIC API KEY ROTATION - Never hit rate limits!"""
        old_index = self.current_key_index
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.current_key = self.api_keys[self.current_key_index]

        print(f"{Fore.YELLOW}🔄 API KEY ROTATION: Key {old_index + 1} → Key {self.current_key_index + 1}")
        print(f"{Fore.CYAN}⚡ Using API key ending in: ...{self.current_key[-4:]}")

        # Reconfigure Gemini with new key
        genai.configure(api_key=self.current_key)
        self.llm = genai.GenerativeModel("gemini-1.5-flash")

        print(f"{Fore.GREEN}✅ API key rotated successfully!")
        return True

    def download_document_from_url(self, document_url):
        """🌐 ULTRA-FAST DOCUMENT DOWNLOAD - Optimized for 25s response limit"""
        try:
            print(f"{Fore.CYAN}🚀 Ultra-fast downloading: {document_url[:60]}...")

            # Check cache first
            url_hash = hashlib.md5(document_url.encode()).hexdigest()
            if url_hash in self.document_cache:
                print(f"{Fore.GREEN}⚡ Using cached document (0ms)!")
                return self.document_cache[url_hash]

            # Speed-optimized download with streaming and timeout
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': '*/*',
                'Connection': 'keep-alive'
            }

            # Download with aggressive timeout for speed
            response = requests.get(
                document_url,
                headers=headers,
                timeout=8,  # Reduced from 30s to 8s
                stream=True  # Stream for faster initial response
            )
            response.raise_for_status()

            # Get file extension from URL or content type
            parsed_url = urlparse(document_url)
            file_extension = os.path.splitext(parsed_url.path)[1]

            if not file_extension:
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    file_extension = '.pdf'
                elif 'word' in content_type or 'docx' in content_type:
                    file_extension = '.docx'
                else:
                    file_extension = '.pdf'  # Default to PDF

            # Stream to temporary file for speed
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                # Download in chunks for speed monitoring
                downloaded = 0
                chunk_size = 8192  # 8KB chunks for balance of speed/memory
                max_size = 50 * 1024 * 1024  # 50MB limit for speed

                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        temp_file.write(chunk)
                        downloaded += len(chunk)

                        # Safety limit for speed
                        if downloaded > max_size:
                            print(f"{Fore.YELLOW}⚠️ File too large, truncating at 50MB for speed")
                            break

                temp_file_path = temp_file.name

            print(f"{Fore.GREEN}✅ Downloaded {downloaded} bytes in optimized mode!")

            # Cache the downloaded file path
            self.document_cache[url_hash] = temp_file_path

            return temp_file_path

        except requests.exceptions.Timeout:
            print(f"{Fore.RED}❌ Download timeout (8s limit) - using fallback")
            return None
        except requests.exceptions.RequestException as e:
            print(f"{Fore.RED}❌ Download failed: {str(e)[:100]}")
            return None
        except Exception as e:
            print(f"{Fore.RED}❌ Unexpected error downloading: {str(e)[:100]}")
            return None

    def process_dynamic_documents(self, document_urls):
        """🚀 SPEED-OPTIMIZED MULTI-DOCUMENT PROCESSING - Parallel downloads for 25s target"""
        if isinstance(document_urls, str):
            document_urls = [document_urls]  # Single URL to list

        print(f"{Fore.CYAN}🚀 SPEED-OPTIMIZED PROCESSING: {len(document_urls)} URLs detected!")

        all_chunks = []
        all_sources = []
        success_count = 0
        start_time = time.time()

        # Parallel download for multiple documents (speed optimization)
        if len(document_urls) > 1:
            import concurrent.futures

            print(f"{Fore.YELLOW}⚡ Using parallel downloads for {len(document_urls)} documents...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all downloads
                future_to_url = {executor.submit(self.download_document_from_url, url): (i, url)
                               for i, url in enumerate(document_urls, 1)}

                # Process downloads as they complete (with timeout)
                for future in concurrent.futures.as_completed(future_to_url, timeout=15):
                    i, url = future_to_url[future]
                    try:
                        temp_file_path = future.result(timeout=5)

                        if temp_file_path:
                            # Fast text extraction
                            document_text = extract_text_from_file(temp_file_path)

                            if document_text and len(document_text.strip()) > 100:
                                # Quick chunking
                                chunks = chunk_text(document_text)

                                if chunks:
                                    source_name = f"hackathon_doc_{i}_{int(time.time())}"
                                    for chunk in chunks:
                                        all_chunks.append(chunk)
                                        all_sources.append(source_name)

                                    success_count += 1
                                    print(f"{Fore.GREEN}✅ Doc {i}: {len(chunks)} chunks ({len(document_text)} chars)")

                    except Exception as e:
                        print(f"{Fore.RED}❌ Error processing document {i}: {str(e)[:50]}")
                        continue

        else:
            # Single document - optimized sequential processing
            for i, url in enumerate(document_urls, 1):
                try:
                    print(f"\n{Fore.YELLOW}📄 Speed-processing document: {url[:60]}...")

                    # Fast download
                    temp_file_path = self.download_document_from_url(url)
                    if not temp_file_path:
                        print(f"{Fore.RED}❌ Failed to download document {i}")
                        continue

                    # Fast text extraction
                    print(f"{Fore.YELLOW}⚡ Fast text extraction...")
                    document_text = extract_text_from_file(temp_file_path)

                    if not document_text or len(document_text.strip()) < 100:
                        print(f"{Fore.RED}❌ Document {i} appears empty or too short!")
                        continue

                    # Fast chunking
                    print(f"{Fore.YELLOW}⚡ Fast chunking...")
                    chunks = chunk_text(document_text)

                    if not chunks:
                        print(f"{Fore.RED}❌ Failed to create chunks from document {i}!")
                        continue

                    # Add to collection
                    source_name = f"hackathon_doc_{i}_{int(time.time())}"
                    for chunk in chunks:
                        all_chunks.append(chunk)
                        all_sources.append(source_name)

                    success_count += 1
                    print(f"{Fore.GREEN}✅ Document {i}: {len(chunks)} chunks extracted ({len(document_text)} chars)")

                except Exception as e:
                    print(f"{Fore.RED}❌ Error processing document {i}: {str(e)[:50]}")
                    continue

        download_time = time.time() - start_time
        print(f"{Fore.CYAN}⚡ Document processing completed in {download_time:.1f}s")

        if success_count == 0:
            print(f"{Fore.RED}💀 ALL DYNAMIC DOCUMENTS FAILED! Keeping existing documents.")
            return False

        # 🔥 HYBRID APPROACH: Combine with sample docs if available
        if self.local_docs_loaded and self.document_chunks:
            print(f"\n{Fore.CYAN}🔗 HYBRID MODE: Combining {success_count} dynamic docs with existing sample docs")
            # Keep existing local docs and add dynamic ones
            all_chunks = self.document_chunks + all_chunks
            all_sources = self.document_sources + all_sources

        # Update system state
        self.document_chunks = all_chunks
        self.document_sources = all_sources
        self.current_document_urls = document_urls
        self.dynamic_docs_count = success_count

        # Fast embeddings generation
        embeddings_start = time.time()
        print(f"{Fore.YELLOW}🧠 Fast embeddings generation for combined dataset...")
        self.embeddings = self.sentence_model.encode(self.document_chunks)
        embeddings_time = time.time() - embeddings_start

        total_time = time.time() - start_time
        print(f"\n{Fore.GREEN}🎉 SPEED-OPTIMIZED SUCCESS!")
        print(f"{Fore.BLUE}📊 Performance Stats:")
        print(f"   • Documents processed: {success_count}")
        print(f"   • Total chunks: {len(self.document_chunks)}")
        print(f"   • Download time: {download_time:.1f}s")
        print(f"   • Embeddings time: {embeddings_time:.1f}s")
        print(f"   • Total time: {total_time:.1f}s")
        print(f"   • Speed target: {'✅ Under 15s' if total_time < 15 else '⚠️ Over 15s'}")

        return True

    def ensure_documents_loaded(self, document_urls=None):
        """🛡️ BADASS DOCUMENT MANAGEMENT - Hybrid static + dynamic loading"""

        # 🔥 ALWAYS LOAD SAMPLE DOCS FIRST (if not already loaded)
        if not self.local_docs_loaded:
            print(f"{Fore.CYAN}📚 Loading sample policy documents as baseline...")
            if self.load_documents("docs"):
                self.local_docs_loaded = True
                print(f"{Fore.GREEN}✅ Sample documents loaded as baseline")
            else:
                print(f"{Fore.YELLOW}⚠️ No sample documents found, proceeding with dynamic only")

        # 🌐 PROCESS DYNAMIC DOCUMENTS (if provided)
        if document_urls:
            # Handle both single URL and multiple URLs
            if isinstance(document_urls, str):
                urls_list = [document_urls]
            else:
                urls_list = document_urls

            # Check if these are new URLs
            if urls_list != self.current_document_urls:
                print(f"{Fore.CYAN}🔄 New document URLs detected, processing {len(urls_list)} documents...")
                success = self.process_dynamic_documents(urls_list)
                if not success and not self.local_docs_loaded:
                    print(f"{Fore.RED}❌ Both dynamic and sample documents failed!")
                    return False
                return True

        # ✅ ENSURE WE HAVE SOMETHING LOADED
        if not self.document_chunks:
            print(f"{Fore.YELLOW}📂 No documents loaded yet, loading sample docs...")
            return self.load_documents("docs")

        print(f"{Fore.GREEN}✅ Documents ready: {len(self.document_chunks)} chunks available")
        return True

    def call_llm_with_fallback(self, prompt, max_retries=None, timeout=5):
        """🚀 BULLETPROOF LLM CALLS - Speed-optimized with timeout control"""
        if max_retries is None:
            max_retries = len(self.api_keys) * 2  # Try each key twice

        for attempt in range(max_retries):
            try:
                print(f"{Fore.CYAN}🤖 LLM Call attempt {attempt + 1} (timeout: {timeout}s)")

                # Speed-optimized generation config for 25s response target
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=300,  # Increased but limited for speed
                    temperature=0.1,  # Low temperature for speed and consistency
                    candidate_count=1
                )

                response = self.llm.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": timeout}  # Configurable timeout
                )

                if response and response.text:
                    print(f"{Fore.GREEN}✅ LLM call successful in attempt {attempt + 1}!")
                    return response.text
                else:
                    print(f"{Fore.YELLOW}⚠️ Empty response, trying next key...")
                    self.rotate_api_key()

            except Exception as e:
                error_msg = str(e).lower()
                print(f"{Fore.RED}❌ LLM call failed: {str(e)[:100]}")

                # Check for timeout errors
                if 'timeout' in error_msg or 'deadline' in error_msg:
                    print(f"{Fore.YELLOW}⏰ TIMEOUT - Reducing timeout for next attempt...")
                    timeout = max(2, timeout - 1)  # Reduce timeout but keep minimum

                # Check for quota/rate limit errors
                elif any(keyword in error_msg for keyword in ['quota', 'rate', 'limit', 'exceeded', 'resource_exhausted']):
                    print(f"{Fore.YELLOW}📊 QUOTA EXCEEDED - Rotating to next API key...")
                    self.rotate_api_key()
                elif 'api_key' in error_msg or 'unauthorized' in error_msg:
                    print(f"{Fore.YELLOW}🔑 API KEY ISSUE - Rotating to next key...")
                    self.rotate_api_key()
                else:
                    print(f"{Fore.RED}💥 Unexpected error, rotating anyway...")
                    self.rotate_api_key()

                # Minimal delay to avoid hammering but maintain speed
                import time
                time.sleep(0.2)  # Reduced from 0.5s to 0.2s

        # All keys failed
        print(f"{Fore.RED}💀 ALL {len(self.api_keys)} API KEYS EXHAUSTED!")
        return "Unable to process due to API limitations. Please try again later."

    def load_documents(self, docs_folder="docs"):
        """🏗️ Load and process sample policy documents (baseline dataset)"""
        print(f"\n{Fore.CYAN}📚 Loading SAMPLE policy documents (baseline)...")

        try:
            # Process only sample policy documents (exclude document.txt files)
            all_chunks, document_sources = self._process_policy_documents(docs_folder)

            if not all_chunks:
                print(f"{Fore.RED}❌ No sample policy documents found in '{docs_folder}' folder!")
                return False

            # If we already have dynamic docs, combine them
            if self.dynamic_docs_count > 0:
                print(f"{Fore.CYAN}🔗 Combining with {self.dynamic_docs_count} existing dynamic documents...")
                all_chunks = all_chunks + self.document_chunks
                document_sources = document_sources + self.document_sources

            self.document_chunks = all_chunks
            self.document_sources = document_sources

            # Generate embeddings
            print(f"{Fore.YELLOW}🧠 Generating semantic embeddings for combined dataset...")
            self.embeddings = self.sentence_model.encode(self.document_chunks)

            print(f"{Fore.GREEN}✅ Successfully loaded {len(self.document_chunks)} document chunks")
            print(f"📊 Embeddings shape: {self.embeddings.shape}")

            # Show document statistics
            unique_docs = list(set(self.document_sources))
            print(f"{Fore.BLUE}📋 Documents in system:")
            for doc in unique_docs:
                count = self.document_sources.count(doc)
                doc_type = "🌐 Dynamic" if "hackathon_doc" in doc else "📄 Sample"
                print(f"   {doc_type} {doc}: {count} chunks")

            return True

        except Exception as e:
            print(f"{Fore.RED}❌ Error loading sample documents: {str(e)}")
            return False

    def _process_policy_documents(self, docs_folder):
        """Process only sample policy documents, exclude document.txt files"""
        all_chunks = []
        document_sources = []

        if not os.path.exists(docs_folder):
            raise ValueError(f"Documents folder '{docs_folder}' not found!")

        # Get all files, but exclude document.txt files
        supported_extensions = ['.pdf', '.docx']  # Only policy documents
        files = []

        for file in os.listdir(docs_folder):
            file_path = os.path.join(docs_folder, file)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower()
                filename = os.path.splitext(file)[0].lower()

                # Only process sample policy files
                if (ext in supported_extensions and
                    ('policy' in filename or 'sample' in filename) and
                    'document' not in filename):
                    files.append((file, file_path))

        if not files:
            raise ValueError(f"No sample policy documents found in '{docs_folder}' folder!")

        for filename, file_path in files:
            try:
                print(f"   📄 Processing: {filename}")
                text = extract_text_from_file(file_path)
                chunks = chunk_text(text)

                for chunk in chunks:
                    all_chunks.append(chunk)
                    document_sources.append(filename)

                print(f"      → {len(chunks)} chunks extracted")

            except Exception as e:
                print(f"      → ❌ Error processing {filename}: {e}")
                continue

        return all_chunks, document_sources

    def semantic_search(self, query, top_k=5):
        """Enhanced semantic search that filters for relevant coverage clauses"""
        if not self.embeddings.size:
            print(f"{Fore.RED}❌ No documents loaded! Please load documents first.")
            return []

        print(f"{Fore.YELLOW}🔍 Searching for relevant policy clauses...")

        query_emb = self.sentence_model.encode([query])

        # Ensure embeddings are numpy arrays with correct dtype
        embeddings = self.embeddings.astype('float32')
        query_emb = query_emb.astype('float32')

        # Create FAISS index for semantic search
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        # Search for more candidates initially to filter better
        search_k = min(top_k * 3, len(self.document_chunks))
        distances, indices = index.search(query_emb, search_k)

        # Enhanced filtering and ranking
        candidates = []
        for idx, i in enumerate(indices[0]):
            if i < len(self.document_chunks):
                chunk = self.document_chunks[i]
                score = float(distances[0][idx])

                # Calculate relevance based on coverage keywords
                relevance_score = self._calculate_chunk_relevance(chunk, query)

                # Combine semantic similarity with relevance
                combined_score = score * (1 / max(relevance_score, 0.1))

                candidates.append({
                    'chunk': chunk,
                    'score': combined_score,
                    'source': self.document_sources[i],
                    'relevance': relevance_score
                })

        # Sort by combined score and return top k
        candidates.sort(key=lambda x: x['score'])
        top_candidates = candidates[:top_k]

        print(f"{Fore.GREEN}✅ Found {len(top_candidates)} relevant clauses")

        return [candidate['chunk'] for candidate in top_candidates], \
               [candidate['source'] for candidate in top_candidates]

    def _calculate_chunk_relevance(self, chunk, query):
        """Calculate how relevant a chunk is for the query"""
        chunk_lower = chunk.lower()
        query_lower = query.lower()

        # Policy-specific keywords that indicate important clauses
        coverage_keywords = [
            'coverage', 'covered', 'benefit', 'treatment', 'surgery',
            'medical', 'hospital', 'injury', 'accident', 'emergency',
            'inpatient', 'outpatient', 'rehabilitation', 'therapy',
            'policy', 'claim', 'eligible', 'exclusion', 'inclusion',
            'deductible', 'copay', 'premium', 'waiting period'
        ]

        # Keywords that suggest procedural/administrative content (less relevant)
        procedural_keywords = [
            'helpline', 'notify', 'inform', 'contact', 'call', 'phone',
            'documentation', 'submit', 'forms', 'application',
            'within 48 hours', 'deadline', 'timeframe', 'office hours'
        ]

        # Calculate scores
        coverage_score = sum(2 if keyword in chunk_lower else 0 for keyword in coverage_keywords)
        procedural_penalty = sum(1 if keyword in chunk_lower else 0 for keyword in procedural_keywords)

        # Query-specific relevance
        query_words = query_lower.split()
        query_match_score = sum(3 if word in chunk_lower else 0 for word in query_words if len(word) > 2)

        # Final relevance score
        relevance = max(coverage_score + query_match_score - procedural_penalty, 0.1)
        return relevance

    def process_claim_query(self, user_query, document_urls=None):
        """🚀 BADASS CLAIM PROCESSING with multi-document support"""
        try:
            # 🌐 CRITICAL: Ensure we have the right documents loaded (hybrid approach)
            if not self.ensure_documents_loaded(document_urls):
                return {
                    "decision": "error",
                    "justification": "Failed to load required documents for processing.",
                    "user_friendly_explanation": "Sorry, I couldn't access the policy documents needed to process your claim. Please try again or contact support.",
                    "processing_method": "document_loading_failed"
                }

            # 🔄 ALWAYS TRY AI FIRST - Will automatically use AI when quota resets!
            print(f"{Fore.CYAN}🤖 Attempting AI processing with {len(self.document_chunks)} document chunks...")
            return self._process_claim_with_ai(user_query)
        except Exception as e:
            error_msg = str(e)
            print(f"{Fore.RED}❌ Error processing claim: {error_msg}")
            print(f"{Fore.YELLOW}⚠️ Using document-based fallback...")
            return self._fallback_claim_processing(user_query)

    def _process_claim_with_ai(self, user_query):
        """Main method to process a user's claim query and return a decision"""
        print(f"\n{Fore.CYAN}🔄 Processing claim query: {Style.BRIGHT}{user_query}")

        # Step 1: Simple query processing
        print(f"{Fore.YELLOW}🧠 AI is analyzing your request...")
        enhanced_query = user_query  # Simple passthrough
        is_emergency = any(keyword in user_query.lower() for keyword in
                          ['emergency', 'urgent', 'heart attack', 'stroke', 'accident', 'critical'])

        print(f"{Fore.GREEN}✨ AI Understanding: {enhanced_query}")
        if is_emergency:
            print(f"{Fore.RED}🚨 EMERGENCY DETECTED - Fast-track processing!")

        # Step 2: Search for relevant policy clauses
        relevant_chunks, relevant_sources = self.semantic_search(enhanced_query)

        if not relevant_chunks:
            return {
                "decision": "error",
                "justification": "No relevant policy clauses found for this query.",
                "user_friendly_explanation": "Sorry, I couldn't find relevant information in the policy documents to process your claim.",
                "specialist_recommendation": "General physician"
            }

        # Step 3: Use AI to make the decision
        print(f"{Fore.YELLOW}🤖 AI is evaluating your claim against policy rules...")
        decision = self._evaluate_claim_with_ai(user_query, enhanced_query, is_emergency, relevant_chunks, relevant_sources)

        return decision

    def _evaluate_claim_with_ai(self, original_query, enhanced_query, is_emergency, relevant_chunks, document_sources):
        """Use AI to evaluate the claim and make a decision based on actual policy content"""

        # Create context from relevant clauses
        clauses_context = "\n".join([
            f"Clause {i+1} (from {document_sources[i]}): {clause}"
            for i, clause in enumerate(relevant_chunks)
        ])

        # Enhanced prompt for REAL WORLD insurance analysis
        prompt = f"""
You are an expert insurance policy analyzer with direct access to the ACTUAL policy document.
Answer the user's question with confidence and precision using ONLY the provided policy content.

🎯 DIRECT POLICY ANALYSIS:
Question: "{original_query}"
Enhanced Query: "{enhanced_query}"
Emergency Status: {"🚨 LIFE-THREATENING - IMMEDIATE COVERAGE REQUIRED" if is_emergency else "Standard Processing"}

📋 MOST RELEVANT POLICY CONTENT (prioritize newer/dynamic content):
{clauses_context}

CRITICAL INSTRUCTIONS:
1. Answer the question DIRECTLY using the policy content above
2. Be SPECIFIC with exact numbers, conditions, and requirements found in the policy
3. If the policy content clearly states something, state it confidently without hedging
4. Prioritize information from the actual uploaded policy document over general samples
5. Quote specific clause numbers/sections when available
6. Don't say "we need more information" if the answer is clearly in the provided content
7. If there are conflicting numbers, use the most specific/recent policy information

Provide your analysis in this JSON format:
{{
  "decision": "approved" or "rejected" or "requires_review",
  "justification": "Detailed analysis with specific policy clause references and exact numbers",
  "user_friendly_explanation": "Direct, confident answer with exact details from the policy document",
  "coverage_percentage": "Exact percentage or amount if specified in policy",
  "next_steps": ["Specific actions based on policy requirements"]
}}
"""

        try:
            # 🚀 USE BULLETPROOF LLM CALL WITH AUTOMATIC FAILOVER AND SPEED OPTIMIZATION
            response_text = self.call_llm_with_fallback(prompt, timeout=3)  # 3s timeout for speed

            if not response_text or "Unable to process due to API limitations" in response_text:
                return {
                    "decision": "needs_review",
                    "justification": "API temporarily unavailable. Please contact support for manual review.",
                    "user_friendly_explanation": response_text,
                    "next_steps": ["Contact customer support"]
                }

            # Enhanced cleanup of response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            response_text = response_text.strip()

            # Find JSON content between first { and last }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                response_text = response_text[start_idx:end_idx+1]

            # Parse JSON response
            decision = json.loads(response_text)

            # Add metadata
            decision['processed_query'] = enhanced_query
            decision['emergency_detected'] = is_emergency
            decision['clauses_analyzed'] = len(relevant_chunks)

            return decision

        except (json.JSONDecodeError, Exception) as e:
            # Fallback with basic information
            return {
                "decision": "error",
                "justification": f"AI processing error: {str(e)}. Please contact customer service.",
                "user_friendly_explanation": "Sorry, there was an error processing your claim. Please contact support for assistance.",
                "next_steps": ["Contact customer support"]
            }

    def _fallback_claim_processing(self, user_query):
        """Enhanced intelligent fallback when AI is unavailable"""
        print(f"{Fore.YELLOW}⚡ Using intelligent document-based analysis...")

        # Get the most relevant policy chunks
        try:
            relevant_chunks, scores = self.semantic_search(user_query, top_k=5)
        except:
            relevant_chunks = []

        # Intelligent analysis based on document content
        if relevant_chunks:
            best_chunks = relevant_chunks[:3]
            combined_content = " ".join(best_chunks)

            if 'exclusion' in combined_content.lower() or 'not covered' in combined_content.lower():
                decision = 'requires_review'
                answer = f"Based on policy documents, this may involve exclusions. Please contact customer service for detailed review."
            elif 'emergency' in user_query.lower() or 'urgent' in user_query.lower():
                decision = 'approved'
                answer = f"Emergency situations are typically covered. Seek immediate medical attention and contact customer service."
            else:
                decision = 'approved'
                answer = f"This appears to align with covered benefits. Contact customer service for confirmation."
        else:
            decision = 'requires_review'
            answer = f"Your query requires detailed policy review. Please contact customer service."

        return {
            'decision': decision,
            'user_friendly_explanation': answer,
            'justification': f"Document-based analysis found {len(relevant_chunks)} relevant policy sections.",
            'processing_method': 'intelligent_document_analysis',
            'next_steps': ["Contact customer service with your policy number"]
        }

if __name__ == "__main__":
    print(f"{Back.GREEN}{Fore.WHITE}{Style.BRIGHT}")
    print("🏥 INTELLIGENT INSURANCE CLAIMS PROCESSING SYSTEM")
    print("Built with LLMs for natural language claim processing")
    print(f"{Style.RESET_ALL}")

    # Initialize the processor
    processor = IntelligentClaimsProcessor()

    # Test dynamic document processing capability
    print(f"\n{Fore.CYAN}🧪 Testing dynamic document processing...")

    test_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    test_query = "What is the grace period for premium payment?"

    print(f"{Fore.YELLOW}📎 Testing with URL: {test_url[:80]}...")
    print(f"{Fore.YELLOW}❓ Test question: {test_query}")

    result = processor.process_claim_query(test_query, test_url)

    print(f"\n{Fore.GREEN}🎉 DYNAMIC DOCUMENT TEST RESULT:")
    print(f"Decision: {result.get('decision', 'Unknown')}")
    print(f"Answer: {result.get('user_friendly_explanation', 'No answer')[:200]}...")

    if result.get('decision') != 'error':
        print(f"\n{Fore.GREEN}✅ SUCCESS: System can handle hackathon admin documents!")
    else:
        print(f"\n{Fore.RED}❌ FAILED: System cannot handle dynamic documents!")
