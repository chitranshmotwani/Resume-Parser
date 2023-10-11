# resume_parser_tool/app/app.py

import streamlit as st
from data_preprocessing.preprocessing import preprocess_text
from feature_extraction.tfidf import extract_tfidf_features
from model.similarity_model import calculate_similarity

# Function to calculate similarity score
def calculate_similarity_score(job_description, resume):
    # Preprocess job description and resume
    preprocessed_job_description = preprocess_text(job_description)
    preprocessed_resume = preprocess_text(resume)

    # Combine job description and resume into a list
    corpus = [preprocessed_job_description, preprocessed_resume]

    # Extract TF-IDF features
    tfidf_matrix = extract_tfidf_features(corpus)

    # Calculate similarity score
    similarity_score = calculate_similarity(tfidf_matrix)[0, 1]

    return similarity_score * 100  # Convert to percentage

# Streamlit application
st.title("Resume Parser Tool")

# Job description input
job_description = st.text_area("Enter the job description:")

# Resume input
resume = st.text_area("Enter the resume:")

if job_description and resume:
    similarity_score = calculate_similarity_score(job_description, resume)

    # Display similarity score
    st.write(f"Similarity Score: {similarity_score:.2f}%")
