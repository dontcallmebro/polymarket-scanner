"""Test Polymarket simplified-markets endpoint."""
import requests
import json

print("Testing Polymarket CLOB simplified-markets API...")
print("=" * 60)

try:
    r = requests.get(
        'https://clob.polymarket.com/simplified-markets',
        params={'closed': 'false'},
        timeout=20
    )
    print(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        markets = data.get('data', [])
        print(f"Markets returned: {len(markets)}")
        print()
        
        # Filter only active non-closed markets with recent activity
        active_markets = [m for m in markets if m.get('active') and not m.get('closed')]
        print(f"Active non-closed: {len(active_markets)}")
        print()
        
        for i, m in enumerate(markets[:10], 1):
            q = m.get('question', 'N/A')
            cond_id = m.get('condition_id', 'N/A')
            active = m.get('active', False)
            closed = m.get('closed', True)
            tokens = m.get('tokens', [])
            
            print(f"{i}. {q[:60]}...")
            print(f"   Active: {active}, Closed: {closed}")
            print(f"   Condition ID: {cond_id[:40]}...")
            if tokens:
                for t in tokens[:2]:
                    print(f"   Token: {t.get('token_id', 'N/A')[:30]}... = {t.get('outcome', 'N/A')}")
            print()
            
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
