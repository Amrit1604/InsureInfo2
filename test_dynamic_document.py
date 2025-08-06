"""
🧪 BADASS MULTI-DOCUMENT TESTING
================================
Test script to verify that the system can handle multiple unknown document sources
from hackathon admins - ULTIMATE HACKATHON STRESS TEST! 🔥
"""

import requests
import json
import time

# Test configuration
API_BASE_URL = "http://localhost:8080"
TEST_DOCUMENT_URL_1 = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

def test_single_document():
    """Test single document processing"""
    
    print("🧪 TEST 1: SINGLE DOCUMENT PROCESSING")
    print("=" * 50)
    
    test_payload = {
        "documents": TEST_DOCUMENT_URL_1,
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
            "Does this policy cover maternity expenses?"
        ]
    }
    
    return run_test(test_payload, "Single Document")

def test_multiple_documents():
    """Test multiple document processing (if we had more URLs)"""
    
    print("\n🧪 TEST 2: MULTI-DOCUMENT SIMULATION")
    print("=" * 50)
    
    # For now, use the same URL twice to simulate multiple documents
    test_payload = {
        "documents": f"{TEST_DOCUMENT_URL_1},{TEST_DOCUMENT_URL_1}",
        "questions": [
            "What are all the covered benefits across all policies?",
            "Compare waiting periods between different policies",
            "What are the premium payment options?"
        ]
    }
    
    return run_test(test_payload, "Multi-Document")

def test_hybrid_mode():
    """Test hybrid mode (sample + dynamic documents)"""
    
    print("\n🧪 TEST 3: HYBRID MODE (Sample + Dynamic)")
    print("=" * 50)
    
    test_payload = {
        "documents": TEST_DOCUMENT_URL_1,
        "questions": [
            "Compare this policy with our sample policies",
            "What are the key differences in coverage?",
            "Which policy offers better emergency coverage?"
        ]
    }
    
    return run_test(test_payload, "Hybrid Mode")

def test_sample_only():
    """Test sample documents only"""
    
    print("\n🧪 TEST 4: SAMPLE DOCUMENTS ONLY")
    print("=" * 50)
    
    test_payload = {
        "documents": "",  # Empty to use sample docs
        "questions": [
            "What is covered under our sample insurance policy?",
            "Are dental procedures covered?",
            "What is the claim process?"
        ]
    }
    
    return run_test(test_payload, "Sample Only")

def run_test(test_payload, test_name):
    """Run a test with the given payload"""
    
    print(f"📋 {test_name} Test Questions:")
    for i, question in enumerate(test_payload["questions"], 1):
        print(f"   {i}. {question}")
    print()
    
    try:
        print("🚀 Sending request to API...")
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE_URL}/hackrx/run",
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=180  # 3 minute timeout for multiple documents
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ SUCCESS: {test_name} test passed!")
            print("=" * 30)
            
            # Analyze the results
            answers = result.get("answers", [])
            successful_count = result.get("successful_count", 0)
            
            print(f"📊 Results Summary:")
            print(f"   • Questions processed: {len(answers)}")
            print(f"   • Successful analyses: {successful_count}")
            print(f"   • Success rate: {(successful_count/len(test_payload['questions'])*100):.1f}%")
            print(f"   • Avg time per question: {(processing_time/len(test_payload['questions'])):.2f}s")
            
            print(f"\n📝 Sample Answer:")
            if answers:
                print(f"   Q: {test_payload['questions'][0][:50]}...")
                print(f"   A: {answers[0][:150]}...")
            
            return True
            
        else:
            print(f"\n❌ FAILED: {response.status_code}")
            print(f"Error details: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: Request took too long (>3 minutes)")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Cannot connect to API")
        print("💡 Make sure the API server is running: python -m uvicorn api_server:app --reload")
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {str(e)}")
        return False

def run_stress_test():
    """Run stress test with rapid requests"""
    
    print("\n🧪 TEST 5: STRESS TEST")
    print("=" * 50)
    
    print("🔥 Running 5 rapid requests to test system stability...")
    
    test_payload = {
        "documents": TEST_DOCUMENT_URL_1,
        "questions": ["What is the coverage amount?"]
    }
    
    success_count = 0
    total_time = 0
    
    for i in range(5):
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}/hackrx/run",
                json=test_payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                success_count += 1
                total_time += (end_time - start_time)
                print(f"   ✅ Request {i+1}: {(end_time - start_time):.2f}s")
            else:
                print(f"   ❌ Request {i+1}: Failed ({response.status_code})")
                
        except Exception as e:
            print(f"   ❌ Request {i+1}: Error - {str(e)}")
    
    print(f"\n📊 Stress Test Results:")
    print(f"   • Success rate: {success_count}/5 ({(success_count/5*100):.1f}%)")
    if success_count > 0:
        print(f"   • Average response time: {(total_time/success_count):.2f}s")
    
    return success_count >= 4  # At least 80% success rate

if __name__ == "__main__":
    print("🏥 BADASS INSUREINFO MULTI-DOCUMENT TESTING")
    print("==========================================")
    print("Testing ultimate hackathon readiness! 🔥")
    print()
    
    # Run all tests
    test_results = []
    
    test_results.append(("Single Document", test_single_document()))
    test_results.append(("Multi-Document", test_multiple_documents()))
    test_results.append(("Hybrid Mode", test_hybrid_mode()))
    test_results.append(("Sample Only", test_sample_only()))
    test_results.append(("Stress Test", run_stress_test()))
    
    # Final summary
    print(f"\n🏆 FINAL TEST RESULTS")
    print("=" * 40)
    
    passed_tests = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed_tests += 1
    
    print(f"\n📊 Overall Score: {passed_tests}/{len(test_results)} ({(passed_tests/len(test_results)*100):.1f}%)")
    
    if passed_tests >= 4:
        print(f"\n🎉 BADASS SUCCESS! System is HACKATHON READY! 🔥")
        print(f"🏆 Can handle multiple documents, hybrid mode, and high load!")
    else:
        print(f"\n⚠️ Some tests failed. System needs optimization!")
        
    print(f"\n💪 Ready to dominate the hackathon!")
