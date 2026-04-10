from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

data_ingestion = DataIngestion()
data_transformation = DataTransformation()

df1, df2 = data_ingestion.run()
print(df1.shape)
print(df2.shape)

df1, df2 = data_transformation.run()
print(df1.shape)
print(df2.shape)
