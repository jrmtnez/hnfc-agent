from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stop_words(text):    
    stop_words = set(stopwords.words('english'))
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w.lower() in stop_words]

    return " ".join(filtered_text)
