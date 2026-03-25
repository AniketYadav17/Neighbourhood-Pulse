# Neighbourhood-Pulse
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

The Neighbourhood Pulse: A Spatiotemporal Gentrification Predictor for London.

## Why this matters?

Gentrification is happening all around London, and the best time to invest in these properties was 6 months ago. 

A major way gentrification is measured is by tracking the change in house prices. But, as silly as it sounds, economic indicators suggest the existence of pre-gentrification signals that can help us invest in properties early on. These are usually the areas flocked by young professionals who desire a safer space with low rental prices. An existing trend is the pop-up of independent cafes around these areas, which can act as an early indicator of gentrification. 
Another worthy early signal, before the warehouses turn into flats, shops into restaurants and industrial units into creative studios, are the applications that are filed 12-24 months in advance before the actual change in geography.

Using these data-points and Land Registry data as ground truth, a predictive model is built to quantify these signals and identify undervalued London neighbourhoods before price growth appears. 

This has direct applications in property investment, retail site selection, and mortgage risk assessment.

## The Data

1. [Planning London Datahub](https://planninglondondatahub.london.gov.uk) - Planning Data about use-of-change in property.

Scraped the Webpage to get Planning data - 46,000+ planning applications across three London boroughs 
(pipeline designed to scale to all 33 boroughs, estimated 2-3 million records)

2. [OSMnx OpenStreetmap Network](https://github.com/gboeing/osmnx) - a collaborative, volunteer-driven, open-source project that provides comprehensive, freely accessible global map data, including detailed road networks, buildings, and geographic features. 

Used to extract the data of the existing cafes in and around Greater London.

3. [Land Registry Data](https://landregistry.data.gov.uk/) - Public data about the 'Price Paid Data' used as Ground truth. 

## Tech Stack

Python - Primary language
Elasticsearch API - Reverse engineered the Planning London Datahub's internal API via Chrome DevTools to access 46,000+ planning records without a documented public API
OSMnx - To interact with OSMnx OpenStreetmap Network API with python
Parquet - Column-based file format. 100x faster to query compared to CSV. Machine-readable unlike CSV which is human readable
Pandas — Data manipulation
PyArrow — Parquet engine
Requests - Used to send POST request to the APIs and parse JSON responses.

## Project Structure 
```
neighbourhood-pulse/
│
├── app/
│   └── app.py                  ← Streamlit web application
│
├── data/
│   ├── raw/                    ← Raw fetched data (not tracked by Git)
│   └── processed/              ← Transformed data ready for modelling
│
├── notebooks/
│   └── 01_eda.ipynb            ← Exploratory data analysis
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py   ← Fetches planning and coffee shop data
│   │   ├── data_transformation.py  ← H3 indexing and feature engineering
│   │   └── model_trainer.py    ← XGBoost model training
│   ├── pipelines/
│   │   ├── train_pipeline.py   ← Orchestrates full training pipeline
│   │   └── predict_pipeline.py ← Serves predictions to the app
│   ├── config.py               ← All constants and configuration
│   ├── exceptions.py           ← Custom exception handling
│   ├── logger.py               ← Timestamped logging setup
│   └── utils.py                ← Shared utilities (save/load objects)
│
├── artifacts/                  ← Trained models and encoders (not tracked by Git)
├── main.py                     ← Pipeline entry point
├── Dockerfile                  ← Container setup (in progress)
├── pyproject.toml              ← Package installation configuration
├── requirements.txt            ← Python dependencies
└── CITATIONS.md                ← Data source attributions
```

## How to Run 

1. Clone the repository
```bash
git clone https://github.com/AniketYadav17/Neighbourhood-Pulse.git
cd neighbourhood-pulse
```

2. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

3. Run data ingestion
```bash
python main.py
```

4. Launch the app (coming soon)
```bash
streamlit run app/app.py
```

## Results  
Coming Soon

## Limitations and Future Work

- Currently, planning data from three boroughs (Newham, Redbridge, Waltham Forest) is extracted. Going forward, all 33 boroughs planning data can be extracted for analysis and prediction. 
- Within Planning data, we can extract details about upcoming cafe shops opening and not just the current ones to get a stronger prediction signal.
- We can scrape the data weekly, use a PostgreSQL Database with PostGIS and automate the ML pipeline to retrain the model and observe data drifts. 

## Citations

Boeing, G. (2025). Modeling and Analyzing Urban Networks and Amenities with OSMnx. Geographical Analysis 57 (4), 567-577. doi:10.1111/gean.70009