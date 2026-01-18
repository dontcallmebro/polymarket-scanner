"""Test script to verify Polymarket API is working."""
import requests
import json

print("Testing Polymarket Gamma API...")
print("=" * 50)

try:
    r = requests.get(
        'https://gamma-api.polymarket.com/markets',
        params={'limit': 5, 'active': 'true', 'order': 'volume24hr'},
        timeout=15
    )
    print(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        print(f"Markets returned: {len(data)}")
        print()
        
        for i, m in enumerate(data[:5], 1):
            q = m.get('question', 'N/A')
            print(f"{i}. {q[:70]}...")
            print(f"   ID: {m.get('id')}")
            print(f"   Slug: {m.get('slug')}")
            print(f"   Volume 24h: ${float(m.get('volume24hr', 0) or 0):,.0f}")
            print(f"   End Date: {m.get('endDate')}")
            print()
    else:
        print(f"Error: {r.text}")
        
except Exception as e:
    print(f"Exception: {e}")
