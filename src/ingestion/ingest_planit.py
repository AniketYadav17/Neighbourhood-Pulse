import requests
import sqlite3
import pandas as pd
import time
import os
import json
import random

# --- CONFIGURATION ---
TARGET_BOROUGHS = ["Hackney"]
# TARGET_BOROUGHS = [
#     "Hackney", "Southwark", "Lambeth", "Tower Hamlets", 
#     "Wandsworth", "Islington", "Camden", "Lewisham"
# ]

YEARS_TO_SCRAPE = [2023, 2024, 2025]
BASE_URL = "https://www.planit.org.uk/api/applics/geojson"   #"https://www.planit.org.uk/api/applics/json"

# --- PATHS ---
RAW_DIR = "data/raw/planning"
DB_PATH = "data/processed/london.db"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# --- HEADERS ---

HEADERS = {
    'User-Agent': 'NeighborhoodPulse/1.0 (Student Portfolio; https://github.com/AniketYadav17/Neighbourhood-Pulse)',
    'Accept': 'application/json'
}

def get_planning_data(borough_name, year):
    """
    Scrapes a specific borough for a specific year using the 'auth' parameter.
    """
    page = 1
    total_records = 0
    
    # Slice by year
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    print(f"--- Starting: {borough_name} ({year}) ---")

    while True:
        params = {
            'auth': borough_name,  #Authority
            'start_date': start_date,
            'end_date': end_date,
            'pg_sz': 100,
            'page': page,
            'compress': 'on',
            #'select': 'uid,description,lat,lng,start_date,app_state,app_type,link' 
        }

        try:
            # Pass the HEADERS here
            r = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=10)
            
            # Rate Limit Handling (429)
            if r.status_code == 429:
                print("Rate Limit Hit. Sleeping 60 seconds...")
                time.sleep(60)
                continue
            
            if r.status_code != 200:
                print(f"Error {r.status_code}: {r.text}")
                break

            data = r.json()
            records = data.get('records', [])
            
            if not records:
                print("   No records returned.")
                break 

            # Save to SQLite
            save_batch_to_sqlite(records)
            
            count = len(records)
            total_records += count
            print(f"   Page {page}: Fetched {count} apps. (Total: {total_records})")

            # Check if we are done (if we got fewer rows than the page size)
            if count < 100:
                break 
            
            page += 1
            
            # Politeness Policy: IMPORTANT since we have no key
            # Sleep longer (2-4 seconds) to stay under the radar
            time.sleep(random.uniform(2.0, 4.0))

        except Exception as e:
            print(f"CRASH: {e}")
            break

def save_batch_to_sqlite(data_list):
    conn = sqlite3.connect(DB_PATH)
    df = pd.DataFrame(data_list)
    
    # Rename columns to match schema
    rename_map = {
        'uid': 'id',
        'link': 'url',
        'start_date': 'date_received',
        'app_state': 'status',
        'app_type': 'type'
    }
    df = df.rename(columns=rename_map)
    
    #Clean Lat/Lng
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
    df = df.dropna(subset=['lat', 'lng'])

    # Append
    df.to_sql("planning_apps", conn, if_exists="append", index=False)
    conn.close()

if __name__ == "__main__":
    # Create Table First
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS planning_apps (
            id TEXT PRIMARY KEY,
            description TEXT,
            lat REAL,
            lng REAL,
            date_received TEXT,
            status TEXT,
            type TEXT,
            url TEXT
        )
    """)
    conn.close()

    # The Loop
    for borough in TARGET_BOROUGHS:
        for year in YEARS_TO_SCRAPE:
            get_planning_data(borough, year)
            # Long pause between boroughs to reset any internal counters they might have
            print("   Cooling down for 10 seconds...")
            time.sleep(10)