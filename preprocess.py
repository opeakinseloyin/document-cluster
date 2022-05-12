import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def preprocessor(text):
    """ This function is used to pre-process text by removing punctuations, stopwords, stemming and tokenizing the text"""

    stop_words = stopwords.words('english')
    stemmer = PorterStemmer()
    text = str(text)
    text = text.lower()
    strip_punctuation = str.maketrans('', '', string.punctuation)
    text = text.translate(strip_punctuation)
    text = word_tokenize(text)
    new_word = " ".join([(stemmer.stem(word)) for word in text if word not in stop_words])
    return new_word