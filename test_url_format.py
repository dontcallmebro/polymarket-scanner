import requests

# Test simple de l'API
r = requests.get('https://gamma-api.polymarket.com/markets', params={
    'limit': 2, 
    'closed': 'false', 
    'active': 'true', 
    'order': 'volume24hr', 
    'ascending': 'false'
})

markets = r.json()
for m in markets:
    print('='*80)
    print('Question:', m.get('question'))
    print('Slug:', m.get('slug'))
    # Check all URL-related fields
    for key in sorted(m.keys()):
        if any(x in key.lower() for x in ['url', 'link', 'slug', 'image', 'icon']):
            print(f'  {key}: {m.get(key)}')
