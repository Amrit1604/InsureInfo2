#!/usr/bin/env python3
"""
Quick document reader to analyze the policy and generate questions
"""

from universal_document_processor import UniversalDocumentProcessor

def read_policy_and_generate_questions():
    print("📄 Reading policy document...")
    
    processor = UniversalDocumentProcessor(speed_tier="ultra")
    
    policy_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    # Download and process the document
    document_text = processor.download_and_process_document(policy_url)
    
    if document_text:
        print(f"✅ Document loaded: {len(document_text)} characters")
        print("\n📋 DOCUMENT PREVIEW:")
        print("=" * 80)
        print(document_text[:2000])  # Show first 2000 characters
        print("=" * 80)
        return document_text
    else:
        print("❌ Failed to load document")
        return None

if __name__ == "__main__":
    read_policy_and_generate_questions()
