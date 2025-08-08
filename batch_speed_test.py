#!/usr/bin/env python3
"""
🚀 BATCH SPEED TEST - Target: Under 25 seconds for 10 questions
"""

import time
from universal_document_processor import UniversalDocumentProcessor

def test_batch_speed():
    print("🚀 BATCH SPEED TEST: 10 Questions Processing")
    print("=" * 60)
    print("🎯 TARGET: Under 25 seconds for 10 questions")
    print("⚡ OPTIMIZATION: Ultra-fast batch processing")
    
    # Initialize processor
    start_time = time.time()
    processor = UniversalDocumentProcessor(speed_tier="ultra")
    init_time = time.time() - start_time
    print(f"⚡ Initialization: {init_time:.2f} seconds")
    
    # Test with 10 questions (simulate real server load)
    test_questions = [
        "What are the main benefits of artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?",
        "What is deep learning?",
        "How does AI help in automation?",
        "What are the risks of AI?",
        "How is AI used in healthcare?",
        "What is natural language processing?",
        "How does computer vision work?",
        "What is the future of AI?"
    ]
    
    test_url = "https://arxiv.org/pdf/2301.08727.pdf"  # AI paper
    
    print(f"\n📋 Testing with: {test_url}")
    print(f"❓ Questions: {len(test_questions)} questions")
    print(f"⏰ Target: < 25 seconds")
    print("-" * 60)
    
    # Start timing
    start_time = time.time()
    
    try:
        results = processor.process_multiple_questions_accurately(test_questions, test_url)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n✅ BATCH SPEED TEST COMPLETE!")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"🎯 Target Met: {'YES ✓' if total_time < 25 else 'NO ✗'}")
        print(f"📊 Speed: {len(test_questions)/total_time:.1f} questions/second")
        print(f"⚡ Performance: {57.72 - total_time:.1f} seconds faster than before")
        
        print(f"\n📋 RESULTS SUMMARY:")
        for i, result in enumerate(results[:3], 1):  # Show first 3
            print(f"{i}. {result[:100]}...")
        
        if total_time < 15:
            print("\n🏆 EXCELLENT! Under 15 seconds - Hackathon ready!")
        elif total_time < 25:
            print("\n🎉 GOOD! Under 25 seconds - Target achieved!")
        else:
            print("\n⚠️  Need more optimization...")
            
        return total_time < 25
        
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n❌ Error after {total_time:.2f} seconds: {e}")
        return False

if __name__ == "__main__":
    success = test_batch_speed()
    if success:
        print("\n🎉 BATCH SPEED TEST PASSED! Server is ready.")
    else:
        print("\n⚠️  Need further optimization...")
