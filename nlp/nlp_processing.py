# nlp_processing.py

import gensim
from gensim.models import Word2Vec
import spacy

def preprocess_text(text):
    # Load the English model
    nlp = spacy.load("en_core_web_sm")
    
    # Process the text
    doc = nlp(text)
    
    # Tokenize and lemmatize
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    
    return tokens

def train_word2vec(sentences, vector_size=100, window=5, min_count=1, workers=4):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

def main():
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing is an exciting field of study."
    ]
    
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Train Word2Vec model
    model = train_word2vec(processed_texts)
    
    # Test the model
    print("Similar words to 'learning':")
    similar_words = model.wv.most_similar("learning", topn=3)
    for word, score in similar_words:
        print(f"{word}: {score}")
    
    # Use spaCy for named entity recognition
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
    print("\nNamed Entities:")
    for ent in doc.ents:
        print(f"{ent.text}: {ent.label_}")

if __name__ == "__main__":
    main()
