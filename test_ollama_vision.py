import requests
import json
import time
import sys
import os

# Set the base URL for the API
BASE_URL = "http://localhost:8000/api"

def test_ollamavision():
    """Test the OllamaVision API endpoints"""
    print("Testing OllamaVision API...")
    
    # 1. Get all cameras with OllamaVision filter enabled
    print("\n1. Getting cameras with OllamaVision filter enabled...")
    response = requests.get(f"{BASE_URL}/ollama-vision/cameras")
    
    if response.status_code == 200:
        cameras = response.json()
        print(f"Found {len(cameras)} cameras with OllamaVision filter enabled:")
        for camera in cameras:
            print(f"  - {camera['name']} (ID: {camera['id']}, Active: {camera['is_active']})")
        
        if len(cameras) == 0:
            print("No cameras found with OllamaVision filter enabled.")
            return
    else:
        print(f"Error getting cameras: {response.status_code} - {response.text}")
        return
    
    # 2. Select the first camera for testing
    camera = cameras[0]
    camera_id = camera['id']
    
    # 3. Test a query
    print(f"\n2. Testing query on camera {camera['name']} (ID: {camera_id})...")
    
    # Wait for the processor to initialize if not active
    if not camera['is_active']:
        print("Camera processor is not active. Waiting for initialization...")
        for i in range(10):
            time.sleep(3)
            response = requests.get(f"{BASE_URL}/ollama-vision/cameras")
            if response.status_code == 200:
                cameras = response.json()
                for cam in cameras:
                    if cam['id'] == camera_id and cam['is_active']:
                        print("Camera processor is now active.")
                        break
                else:
                    print(f"Waiting... ({i+1}/10)")
                    continue
                break
        else:
            print("Camera processor did not become active. Proceeding anyway...")
    
    # Send a test query
    query_data = {
        "camera_id": camera_id,
        "query": "What do you see in this camera feed? Describe the scene in detail."
    }
    
    response = requests.post(f"{BASE_URL}/ollama-vision/query", json=query_data)
    
    if response.status_code == 200:
        result = response.json()
        print("\nQuery successful!")
        print(f"Response: {result['response']}")
    else:
        print(f"Error processing query: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_ollamavision()