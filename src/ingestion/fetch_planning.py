import requests
import pandas as pd
import time
import os

# --- Configuration ---
API_KEY = os.getenv("PLANIT_API_KEY", "") 
BASE_URL = "https://www.planit.org.uk/api/applics/json"

# List of target boroughs
BOROUGHS = [
    "Camden", "Hackney", "Southwark", "Lambeth", "Islington", 
    "Tower Hamlets", "Wandsworth", "Hammersmith and Fulham"
]

START_DATE = "2023-01-01"
END_DATE = "2026-01-01"
OUTPUT_PATH = "data/raw/planning_applications.parquet"

def fetch_borough_planning(borough, start, end):
    print(f"📍 Fetching data for: {borough}")
    
    all_records = []
    page = 1
    page_size = 100 
    
    while True:
        params = {
            'auth': borough,
            'start_date': start,
            'end_date': end,
            'pg_sz': page_size,
            'page': page,
            'compress': 'on',
            # REMOVED 'select' to prevent 400 Error (Let API return default fields)
        }
        
        if API_KEY:
            params['k'] = API_KEY

        try:
            response = requests.get(BASE_URL, params=params)
            
            # --- Better Error Handling ---
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"   ⚠️ Rate limit hit. Sleeping for {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            
            if response.status_code != 200:
                # Print the actual error message from the server
                print(f"   ❌ API Error: {response.status_code} - {response.text}")
                break

            data = response.json()
            records = data.get('records', [])
            total_count = data.get('total', 0)
            
            if not records:
                print(f"   ✓ Finished {borough} (No more records)")
                break
                
            all_records.extend(records)
            print(f"   → Page {page}: Retrieved {len(records)} records ({len(all_records)}/{total_count})")
            
            if len(all_records) >= total_count or len(records) < page_size:
                break
            
            page += 1
            time.sleep(1)
            
        except Exception as e:
            print(f"   ❌ System Error fetching {borough}: {e}")
            break
            
    return all_records

def run():
    full_dataset = []
    
    for borough in BOROUGHS:
        borough_data = fetch_borough_planning(borough, START_DATE, END_DATE)
        full_dataset.extend(borough_data)
        time.sleep(2)
    
    if full_dataset:
        df = pd.DataFrame(full_dataset)
        
        # Save
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"\n✅ Success! Saved {len(df)} planning applications to {OUTPUT_PATH}")
    else:
        print("\n⚠️ No data found.")

if __name__ == "__main__":
    run()