import pandas as pd
import h3
import os

# Configuration
H3_RESOLUTION = 9 # Approx city block size
OUTPUT_FILE = "data/processed/unified_training_data.csv"

def simple_nlp_classifier(text):
    """
    V1 NLP: Simple keyword extraction. 
    In V2, we replace this with TF-IDF or BERT.
    """
    if pd.isna(text): return 'other' # Handle missing descriptions
    text = str(text).lower()
    
    if any(x in text for x in ['extension', 'loft', 'conversion', 'renovation']):
        return 'residential_improvement'
    elif any(x in text for x in ['cafe', 'shop', 'a1', 'a3', 'commercial']):
        return 'commercial_change'
    elif 'flats' in text or 'new build' in text:
        return 'new_development'
    else:
        return 'other'

def run():
    print("🚀 Starting ETL Pipeline...")

    # 1. Load Data
    try:
        df_plan = pd.read_csv("data/raw/planning_mock.csv")
        df_amenity = pd.read_csv("data/raw/amenities_mock.csv")
        df_price = pd.read_csv("data/raw/prices_mock.csv")
    except FileNotFoundError:
        print("❌ Error: Raw data not found. Did you run 'src/ingestion/mock_data_generator.py'?")
        return

    # 2. Feature Engineering: NLP on Planning
    print("   ... Running NLP on Planning descriptions")
    df_plan['category'] = df_plan['description'].apply(simple_nlp_classifier)

    # 3. Spatial Engineering: Assign H3 Index to ALL datasets
    # FIX: Use h3.latlng_to_cell instead of h3.geo_to_h3 (Version 4.0+ syntax)
    get_h3 = lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], H3_RESOLUTION)

    df_plan['h3_index'] = df_plan.apply(get_h3, axis=1)
    df_amenity['h3_index'] = df_amenity.apply(get_h3, axis=1)
    df_price['h3_index'] = df_price.apply(get_h3, axis=1)

    # 4. Aggregation (The Unification Step)
    
    # A. Planning Aggregation (Count by Category per Hexagon)
    # We use pivot_table instead of unstack/groupby for safety against missing categories
    plan_agg = pd.pivot_table(
        df_plan, 
        index='h3_index', 
        columns='category', 
        aggfunc='size', 
        fill_value=0
    )
    # Rename columns for clarity (e.g. 'commercial_change' -> 'count_commercial_change')
    plan_agg.columns = [f"count_{col}" for col in plan_agg.columns]
    
    # B. Amenity Aggregation (Count Indues vs Chains)
    df_amenity['is_indie'] = ~df_amenity['name'].str.contains("Starbucks|Pret|Costa", case=False, na=False)
    amenity_agg = df_amenity.groupby('h3_index')['is_indie'].sum().to_frame(name="indie_cafe_count")

    # C. Price Aggregation (The Target Variable)
    price_agg = df_price.groupby('h3_index')['price'].mean().to_frame(name="avg_house_price")

    # 5. Join Everything
    print("   ... Merging datasets on H3 Index")
    # Start with prices as the "anchor" (we only care about hexagons with price data for the model)
    master_df = price_agg.join(plan_agg, how='left').join(amenity_agg, how='left')
    
    # Fill NaNs (e.g., a hexagon might have house sales but no cafes)
    master_df = master_df.fillna(0)
    
    # Reset index to make h3_index a column
    master_df = master_df.reset_index()

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    master_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Pipeline Complete. Unified data saved to {OUTPUT_FILE}")
    print(master_df.head())

if __name__ == "__main__":
    run()