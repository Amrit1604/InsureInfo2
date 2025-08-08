#!/usr/bin/env python3
"""
Speed Test for Universal Document Processor with 6 API Keys
Target: Under 30 seconds for accurate responses
"""

import time
from universal_document_processor import UniversalDocumentProcessor

def test_speed():
    print("🚀 SPEED TEST: Universal Document Processor with 6 API Keys")
    print("=" * 60)

    # Initialize processor
    start_time = time.time()
    processor = UniversalDocumentProcessor()
    init_time = time.time() - start_time
    print(f"⚡ Initialization: {init_time:.2f} seconds")

    # Test with a medium-sized document
    test_question = "What are the main benefits of artificial intelligence?"
    test_url = "https://arxiv.org/pdf/2301.08727.pdf"  # Small AI paper

    print(f"\n📋 Testing with: {test_url}")
    print(f"❓ Question: {test_question}")
    print(f"⏰ Target: < 30 seconds")
    print("-" * 60)

    # Start timing
    start_time = time.time()

    try:
        # Download and process document first
        document_text = processor.download_and_process_document(test_url)

        # Then answer the question
        result = processor.process_question_with_maximum_accuracy(test_question, document_text)

        end_time = time.time()
        total_time = end_time - start_time

        print(f"\n✅ SPEED TEST COMPLETE!")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"🎯 Target Met: {'YES ✓' if total_time < 30 else 'NO ✗'}")
        print(f"📊 Speed Gain: {40 - total_time:.1f} seconds faster than before")

        print(f"\n💡 Answer Preview:")
        print(f"{result[:200]}...")

        return total_time < 30

    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n❌ Error after {total_time:.2f} seconds: {e}")
        return False

if __name__ == "__main__":
    success = test_speed()
    if success:
        print("\n🎉 SPEED TEST PASSED! Ready for production.")
    else:
        print("\n⚠️  Need more optimization...")
