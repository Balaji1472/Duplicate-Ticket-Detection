"""
Test script for the Duplicate Ticket Detection API.
Run this after starting the API server to test all endpoints.
"""

import requests
import json
from typing import Dict, Any

# API Configuration
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_single_duplicate_check():
    """Test single ticket duplicate checking"""
    print("🎫 Testing Single Duplicate Check...")
    
    # Test with a ticket similar to existing ones
    test_ticket = {
        "ticket_id": "TEST001",
        "text": "My internet connection is not working properly"
    }
    
    response = requests.post(f"{BASE_URL}/check-duplicate", json=test_ticket)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Is Duplicate: {result.get('is_duplicate')}")
    print(f"Matches Found: {len(result.get('matches', []))}")
    
    if result.get('matches'):
        for match in result['matches']:
            print(f"  - {match['ticket_id']}: {match['similarity_score']:.4f}")
    
    print("-" * 50)

def test_batch_duplicate_check():
    """Test batch duplicate checking"""
    print("📦 Testing Batch Duplicate Check...")
    
    batch_tickets = {
        "tickets": [
            {
                "ticket_id": "BATCH001",
                "text": "Internet not working at home"
            },
            {
                "ticket_id": "BATCH002", 
                "text": "Cannot access my account password"
            },
            {
                "ticket_id": "BATCH003",
                "text": "How to contact customer support?"
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/check-batch-duplicates", json=batch_tickets)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Total Tickets: {result.get('total_tickets')}")
    print(f"Duplicates Found: {result.get('duplicates_found')}")
    
    for ticket_result in result.get('results', []):
        print(f"  {ticket_result['ticket_id']}: {'DUPLICATE' if ticket_result['is_duplicate'] else 'UNIQUE'}")
    
    print("-" * 50)

def test_add_ticket():
    """Test adding a new ticket"""
    print("➕ Testing Add Ticket...")
    
    # Use a timestamp to ensure unique ID
    import time
    unique_id = f"NEW{int(time.time())}"
    
    new_ticket = {
        "ticket_id": unique_id,
        "text": "I need help with billing inquiry for my account"
    }
    
    response = requests.post(f"{BASE_URL}/add-ticket", json=new_ticket)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result}")
    
    if response.status_code == 200:
        print(f"✅ Successfully added ticket {unique_id}")
    elif response.status_code == 409:
        print(f"⚠️ Ticket {unique_id} already exists")
    else:
        print(f"❌ Failed to add ticket: {result.get('error', 'Unknown error')}")
    
    print("-" * 50)

def test_stats():
    """Test getting system stats"""
    print("📊 Testing Stats Endpoint...")
    
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    stats = response.json()
    print(f"Total Tickets: {stats.get('total_tickets')}")
    print(f"Model: {stats.get('model_name')}")
    print(f"Threshold: {stats.get('similarity_threshold')}")
    print(f"Embedding Dimension: {stats.get('embedding_dimension')}")
    print("-" * 50)

def test_root_endpoint():
    """Test the root endpoint"""
    print("🏠 Testing Root Endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def run_all_tests():
    """Run all API tests"""
    print("🚀 Starting API Tests...")
    print("=" * 50)
    
    try:
        test_root_endpoint()
        test_health_check()
        test_stats()
        test_single_duplicate_check()
        test_batch_duplicate_check()
        test_add_ticket()
        
        # Test stats again to see the added ticket
        print("📊 Updated Stats After Adding Ticket...")
        test_stats()
        
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server.")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    run_all_tests()