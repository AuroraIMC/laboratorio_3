#Codigo desarrollado por aurora matamoros
from flask import Flask, render_template, request, redirect, url_for, flash
import os
import spacy
import pytextrank
from werkzeug.utils import secure_filename  
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt
from collections import Counter
from transformers import pipeline

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nlp_en = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp_en = spacy.load("en_core_web_sm")
try:
    nlp_es = spacy.load("es_core_news_sm")
except OSError:
    from spacy.cli import download
    download("es_core_news_sm")
    nlp_es = spacy.load("es_core_news_sm")
if "textrank" not in nlp_en.pipe_names:
    nlp_en.add_pipe("textrank")
if "textrank" not in nlp_es.pipe_names:
    nlp_es.add_pipe("textrank")

app = Flask(__name__)

def load_book_text(book_filename: str) -> str:
    path = os.path.join("data", book_filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
def clean_gutenberg_text(text: str) -> str:
    start_re = r"\*{3} START OF (THE|THIS) PROJECT GUTENBERG EBOOK .* \*{3}"
    end_re = r"\*{3} END OF (THE|THIS) PROJECT GUTENBERG EBOOK .* \*{3}"

    start_match = re.search(start_re, text, flags=re.IGNORECASE)
    end_match = re.search(end_re, text, flags=re.IGNORECASE)

    if start_match and end_match:
        text = text[start_match.end():end_match.start()]
    return text.strip()

def basic_text_analysis(language: str, book_filename: str) -> dict:
    text = load_book_text(book_filename)
    text = clean_gutenberg_text(text)
    # 1) Tokenización por oraciones y palabras
    sentences = sent_tokenize(text, language="english" if language == "english" else "spanish")
    words_raw = word_tokenize(text, language="english" if language == "english" else "spanish")

    # 2) Normalización: minúsculas, quitar puntuación
    punct_table = str.maketrans("", "", string.punctuation + "«»“”¡¿")
    words_norm = [w.lower().translate(punct_table) for w in words_raw if w.translate(punct_table)]

    # Stop-words (no se eliminan para el conteo, pero pueden servir después)
    stops = set(stopwords.words("english" if language == "english" else "spanish"))

    # 3) Conteos
    num_sentences = len(sentences)
    num_words_raw = len(words_raw)        
    num_words_norm = len(words_norm)     

    # 4) Resumen simple = primeras 3 oraciones razonables
    filtered = [s for s in sentences if len(s.split()) < 30]
    summary_simple = " ".join(filtered[:3]) if filtered else "Resumen no disponible."

    return {
        "num_sentences": num_sentences,
        "num_words_raw": num_words_raw,
        "num_words_norm": num_words_norm,
        "summary": summary_simple,
    }
#sentimientos 
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
def analyze_sentiment(text: str) -> dict:
    # Para textos muy largos
    short_text = text[:512]  
    results = sentiment_analyzer(short_text)
    label = results[0]['label']
    score = results[0]['score']
    return {
        "label": label,
        "score": score
    }
#tema y personaje 
spacy_models = {
    "english": spacy.load("en_core_web_sm"),
    "spanish": spacy.load("es_core_news_md")
}
def identify_topics_and_characters(language: str, book_filename: str) -> dict:
    text = load_book_text(book_filename)
    nlp = spacy_models[language]
    doc = nlp(text[:100000])  # Limitamos a los primeros 100k caracteres

    # Entidades PERSON
    persons = [ent.text for ent in doc.ents if ent.label_ == "PER" or ent.label_ == "PERSON"]
    most_common_persons = Counter(persons).most_common(5)

    # Sustantivos y nombres propios frecuentes
    nouns = [token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
    most_common_nouns = Counter(nouns).most_common(5)

    return {
        "main_characters": [name for name, _ in most_common_persons],
        "main_topics": [noun for noun, _ in most_common_nouns]
    }
#tarjetas 
def generate_flashcards(language: str, book_filename: str) -> list:
    if language == "english":
        nlp = spacy.load("en_core_web_sm")
    else:
        nlp = spacy.load("es_core_news_sm")

    nlp.max_length = 3_000_000

    if "textrank" not in nlp.pipe_names:
        nlp.add_pipe("textrank")
        
    text = load_book_text(book_filename)
    text = clean_gutenberg_text(text)
    chunk_size = 100_000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    phrase_counter = {}

    for chunk in chunks:
        doc = nlp(chunk)
        for phrase in doc._.phrases[:5]:
            text_key = phrase.text.lower().strip()
            if text_key not in phrase_counter:
                phrase_counter[text_key] = {
                    "score": phrase.rank,
                    "summary": phrase.chunks[0].sent.text.strip() if phrase.chunks else ""
                }
    top_phrases = sorted(phrase_counter.items(), key=lambda x: -x[1]["score"])[:10]

    flashcards = []
    for topic, data in top_phrases:
        flashcards.append({
            "topic": topic,
            "summary": data["summary"]
        })

    return flashcards
# preguntas
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def answer_question_transformer(text, question):
    response = qa_pipeline({
        'context': text,
        'question': question
    })
    return response['answer']

#carga de libros 
BOOKS = {
    'english': [
        'emma.txt',
        'persuasion.txt',
        'sense.txt'
    ],
    'spanish': [
        'don_quijote.txt',
        'divina_comedia.txt',
        'capitan_veneno.txt'
    ]
}



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_book', methods=['POST'])
def select_book():
    language = request.form['language']
    books = BOOKS['english'] if language == 'english' else BOOKS['spanish']
    return render_template('select_book.html', language=language, books=books)

@app.route('/select_analysis', methods=['POST'])
def select_analysis():
    language = request.form['language']
    book = request.form['book']
    return render_template('select_analysis.html', language=language, book=book)

@app.route("/result", methods=["POST"])
def result():
    language = request.form["language"]
    book = request.form["book"]
    analysis_type = request.form.get("analysis_type")

    # Cargar y limpiar el texto solo una vez aquí
    text = load_book_text(book)
    text = clean_gutenberg_text(text)
    #basico 
    if analysis_type == "basic":
        analysis = basic_text_analysis(language, book)
        return render_template(
            "result.html",
            language=language,
            book=book,
            analysis_type="basic", 
            num_sentences=analysis["num_sentences"],
            num_words_raw=analysis["num_words_raw"],
            summary=analysis["summary"],
        )
    #sentimiento 
    elif analysis_type == "sentiment":
        sentiment = analyze_sentiment(text)  
        return render_template(
            "result.html",
            language=language,
            book=book,
            sentiment=sentiment,
            analysis_type="sentiment"
        )
    #tema y personaje
    elif analysis_type == "theme":
        theme_info = identify_topics_and_characters(language, book)
        return render_template(
            "result.html",
            language=language,
            book=book,
            analysis_type="theme",  
            main_characters=theme_info["main_characters"],
            main_topics=theme_info["main_topics"]
    )
    elif analysis_type == "flashcards":
        cards = generate_flashcards(language, book)
        return render_template(
            "result.html",
            language=language,
            book=book,
            flashcards=cards,
            analysis_type=analysis_type
    )
    elif analysis_type == "qa":
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

        context = text 
        return render_template(
            "result.html",
            language=language,
            book=book,
            analysis_type="qa",
            context=context  
        )    
    else:
        return "Tipo de análisis no soportado aún", 400
    
@app.route("/answer", methods=["POST"])
def answer_question():
    question = request.form["question"]
    context = request.form["context"]

    from transformers import pipeline
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    result = qa_pipeline(question=question, context=context)

    return render_template(
        "answer.html",
        question=question,
        answer=result["answer"],
        score=result["score"]
    )

if __name__ == '__main__':
    app.run(debug=True)