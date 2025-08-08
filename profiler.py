#!/usr/bin/env python3
"""
🔍 PROFILER: Find where the 56 seconds are being spent
"""

import time
from universal_document_processor import UniversalDocumentProcessor

def profile_processing():
    print("🔍 PROFILING: Where are the 56 seconds going?")
    print("=" * 60)
    
    overall_start = time.time()
    
    # 1. Initialization
    print("📊 Phase 1: Initialization...")
    start = time.time()
    processor = UniversalDocumentProcessor(speed_tier="ultra")
    print(f"   ⏱️ Initialization: {time.time() - start:.2f}s")
    
    # 2. Document Download & Processing
    print("\n📊 Phase 2: Document Processing...")
    test_url = "https://arxiv.org/pdf/2301.08727.pdf"
    questions = ["What are the main benefits of AI?"] * 3  # Just 3 questions for profiling
    
    start = time.time()
    processor._current_doc_hash = None  # Force reprocessing
    
    # Call the actual method to see where time is spent
    try:
        # Check document processing phase
        document_text = processor.download_and_process_document(test_url)
        print(f"   ⏱️ Document download + text extraction: {time.time() - start:.2f}s")
        
        # Check chunking phase
        start = time.time()
        success = processor.process_document_with_accuracy(document_text)
        print(f"   ⏱️ Chunking + embeddings + indexing: {time.time() - start:.2f}s")
        
        if success:
            # Check question processing phase
            start = time.time()
            results = processor._batch_process_questions_ultra_fast(questions)
            print(f"   ⏱️ Question processing (3 questions): {time.time() - start:.2f}s")
            
            print(f"\n🎯 TOTAL TIME: {time.time() - overall_start:.2f}s")
            print("\n📋 BOTTLENECK ANALYSIS:")
            
            # Estimate for 10 questions
            estimated_10q = (time.time() - overall_start) * (10/3)
            print(f"   📊 Estimated time for 10 questions: {estimated_10q:.2f}s")
            
            if estimated_10q > 25:
                print(f"   ⚠️  Still {estimated_10q - 25:.1f}s over target!")
        else:
            print("   ❌ Document processing failed")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        
    print(f"\n🔍 Profile complete: {time.time() - overall_start:.2f}s total")

if __name__ == "__main__":
    profile_processing()
