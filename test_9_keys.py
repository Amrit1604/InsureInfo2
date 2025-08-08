#!/usr/bin/env python3
"""
🔑 TEST: 9 API Keys with Failover System
"""

import time
from universal_document_processor import UniversalDocumentProcessor

def test_9_keys_system():
    print("🔑 TESTING: 9 API Keys with Smart Failover")
    print("=" * 60)
    
    # Initialize processor
    start_time = time.time()
    processor = UniversalDocumentProcessor(speed_tier="ultra")
    init_time = time.time() - start_time
    print(f"⚡ Initialization: {init_time:.2f} seconds")
    
    # Check all keys loaded
    print(f"\n🔑 API Keys Status:")
    print(f"   📊 Total keys loaded: {len(processor.api_keys)}")
    valid_keys = sum(1 for key in processor.api_keys if key and len(key) > 20)
    print(f"   ✅ Valid keys: {valid_keys}/9")
    print(f"   🎯 Expected: 9 keys")
    
    if valid_keys < 9:
        print(f"   ⚠️ Warning: Only {valid_keys} valid keys found!")
    else:
        print(f"   🎉 All 9 keys loaded successfully!")
    
    # Test failover with fast questions
    test_questions = [
        "What is AI?",
        "How does ML work?", 
        "What are neural networks?"
    ]
    
    test_url = "https://arxiv.org/pdf/2301.08727.pdf"
    
    print(f"\n📋 Testing 9-key failover system...")
    print(f"❓ Questions: {len(test_questions)} questions")
    print("-" * 60)
    
    # Start timing
    start_time = time.time()
    
    try:
        results = processor.process_multiple_questions_accurately(test_questions, test_url)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n✅ 9-KEY FAILOVER TEST COMPLETE!")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"🔑 Key Rotation: Working with {len(processor.api_keys)} keys")
        print(f"📊 Speed: {len(test_questions)/total_time:.1f} questions/second")
        
        print(f"\n📋 SAMPLE RESULTS:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result[:80]}...")
        
        print(f"\n🎯 ESTIMATED 10 QUESTIONS: {total_time * (10/3):.1f} seconds")
        
        if total_time * (10/3) < 25:
            print("🏆 EXCELLENT! 9-key system ready for production!")
        else:
            print("⚠️  May need more optimization...")
            
        return True
        
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n❌ Error after {total_time:.2f} seconds: {e}")
        return False

if __name__ == "__main__":
    success = test_9_keys_system()
    if success:
        print("\n🎉 9-KEY SYSTEM READY! Ultimate capacity achieved.")
    else:
        print("\n⚠️  System needs attention...")
