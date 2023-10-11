from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(tfidf_matrix):
    # Calculate cosine similarity between job descriptions and resumes
    similarity_scores = cosine_similarity(tfidf_matrix[:2])  # Assuming first two rows are job descriptions

    return similarity_scores
