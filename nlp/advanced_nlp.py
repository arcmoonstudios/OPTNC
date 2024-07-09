
# root/nlp/advanced_nlu.py
# Implements advanced natural language understanding tasks

from transformers import pipeline

class AdvancedNLU:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.ner = pipeline("ner")
        self.qa_model = pipeline("question-answering")
        self.summarizer = pipeline("summarization")
        self.translator = pipeline("translation_en_to_fr")
        self.text_generator = pipeline("text-generation")

    def analyze_sentiment(self, text):
        return self.sentiment_analyzer(text)[0]

    def extract_entities(self, text):
        return self.ner(text)

    def answer_question(self, context, question):
        return self.qa_model(question=question, context=context)

    def summarize_text(self, text, max_length=150):
        return self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]

    def translate_text(self, text):
        return self.translator(text)[0]['translation_text']

    def generate_text(self, prompt, max_length=50):
        return self.text_generator(prompt, max_length=max_length, num_return_sequences=1)[0]['generated_text']

if __name__ == '__main__':
    nlu = AdvancedNLU()
    
    text = "I love natural language processing! It's fascinating and powerful."
    print("Sentiment:", nlu.analyze_sentiment(text))
    print("Entities:", nlu.extract_entities(text))
    print("Summary:", nlu.summarize_text(text))
    print("Translation:", nlu.translate_text(text))
    print("Generated text:", nlu.generate_text("NLP is"))
