from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_features(corpus):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    return tfidf_matrix.toarray()
