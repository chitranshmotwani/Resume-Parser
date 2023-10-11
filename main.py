# resume_parser_tool/app/main.py

from data_preprocessing.preprocessing import preprocess_text
from feature_extraction.tfidf import extract_tfidf_features
from model.similarity_model import calculate_similarity

def calculate_similarity_scores(job_description, resumes):
    # Preprocess job description and resumes
    preprocessed_job_description = preprocess_text(job_description)
    preprocessed_resumes = [preprocess_text(resume) for resume in resumes]

    # Combine job description and resumes into a single list
    corpus = [preprocessed_job_description] + preprocessed_resumes

    # Extract TF-IDF features
    tfidf_matrix = extract_tfidf_features(corpus)

    # Calculate similarity scores
    similarity_scores = calculate_similarity(tfidf_matrix)

    return similarity_scores

# Example usage
if __name__ == "__main__":
    # Sample job description and resumes
    job_description = "Looking for a software engineer with Python skills."
    resumes = [
        "Experienced software engineer specializing in Python development.",
        "Machine learning enthusiast with Python expertise."
    ]

    # Calculate similarity scores
    similarity_scores = calculate_similarity_scores(job_description, resumes)

    # Display similarity scores
    print("Similarity Scores:")
    print(similarity_scores)