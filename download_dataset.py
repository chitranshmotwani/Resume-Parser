import requests
import os

# Replace with the actual dataset URL
dataset_url = "https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner/download?datasetVersionNumber=1"  # Replace with the actual JSON dataset URL

# Define the directory to save the dataset
data_dir = "./data"

# Create the directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Download the dataset
print("Downloading the dataset...")
response = requests.get(dataset_url)
with open(os.path.join(data_dir, "dataset.json"), "wb") as file:
    file.write(response.content)

print("Dataset downloaded and saved.")
