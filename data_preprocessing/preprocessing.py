import string
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation and special characters
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Join the words back into a single string
    processed_text = ' '.join(words)

    return processed_text
