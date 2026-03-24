from src.components.data_ingestion import DataIngestion

data_ingestion = DataIngestion()
df1, df2 = data_ingestion.run()
print(df1.shape)
print(df2.shape)