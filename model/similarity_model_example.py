# resume_parser_tool/model/similarity_model_example.py

from feature_extraction.tfidf import extract_tfidf_features
from model.similarity_model import calculate_similarity

# Example usage to train the model and evaluate

# Sample job descriptions and resumes
job_descriptions = ["Looking for a software engineer with Python skills.", "Data scientist with experience in machine learning."]
resumes = ["Experienced software engineer specializing in Python development.", "Machine learning enthusiast with Python expertise."]

# Combine job descriptions and resumes into a single list
corpus = job_descriptions + resumes

# Extract TF-IDF features
tfidf_matrix = extract_tfidf_features(corpus)

# Calculate similarity scores
similarity_scores = calculate_similarity(tfidf_matrix)

# Display similarity scores
print("Similarity Scores:")
print(similarity_scores)
