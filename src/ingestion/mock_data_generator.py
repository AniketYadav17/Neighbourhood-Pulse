import pandas as pd
import numpy as np
import os

# Create directories
os.makedirs("data/raw", exist_ok=True)

# 1. Mock Planning Applications (The "Intent")
# We fake text descriptions to test the NLP logic later
planning_data = {
    "planning_id": range(100, 110),
    "date_received": pd.date_range(start="2025-01-01", periods=10),
    "latitude": np.random.uniform(51.520, 51.525, 10),  # Shoreditch area
    "longitude": np.random.uniform(-0.075, -0.070, 10),
    "description": [
        "Erection of a single storey rear extension", # Residential
        "Change of use from A1 (Retail) to A3 (Cafe)", # Commercial/Gentrification
        "Loft conversion with rear dormer", # Residential
        "Installation of new shop front and signage", # Commercial
        "Construction of 3 residential flats", # New Build
        "Basement excavation and extension", # High-value Reno
        "Display of internally illuminated fascia sign", # Commercial
        "Single storey side extension", # Residential
        "Change of use to hot food takeaway", # Commercial
        "Erection of garden studio for home office" # Residential
    ]
}
df_plan = pd.DataFrame(planning_data)
df_plan.to_csv("data/raw/planning_mock.csv", index=False)
print("✅ Generated 10 Planning Apps")

# 2. Mock Amenities (The "Vibe")
amenity_data = {
    "name": ["Monmouth Coffee", "Pret A Manger", "Indie Yoga", "Craft Beer Co", "Starbucks", "Sourdough Bakery"],
    "amenity": ["cafe", "cafe", "studio", "pub", "cafe", "bakery"],
    "latitude": np.random.uniform(51.520, 51.525, 6),
    "longitude": np.random.uniform(-0.075, -0.070, 6)
}
df_amenity = pd.DataFrame(amenity_data)
df_amenity.to_csv("data/raw/amenities_mock.csv", index=False)
print("✅ Generated 6 Amenities")

# 3. Mock House Prices (The "Ground Truth")
price_data = {
    "price": [450000, 520000, 600000, 480000, 750000, 800000, 550000, 490000],
    "date": pd.date_range(start="2025-06-01", periods=8), # Future dates relative to planning
    "latitude": np.random.uniform(51.520, 51.525, 8),
    "longitude": np.random.uniform(-0.075, -0.070, 8)
}
df_price = pd.DataFrame(price_data)
df_price.to_csv("data/raw/prices_mock.csv", index=False)
print("✅ Generated 8 House Transactions")