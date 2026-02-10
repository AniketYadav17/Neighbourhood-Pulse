install:
	pip install -r requirements.txt

data:
	python -m src.ingestion.fetch_planning
	python -m src.ingestion.fetch_amenities
	python -m src.ingestion.fetch_prices

clean:
	rm -rf data/raw/*
	rm -rf data/processed/*