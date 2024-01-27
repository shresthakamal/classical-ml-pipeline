import re
import string
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# 1. lower case
def lower_case(text):
    return text.lower()


# 2. remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


# 3. remove stop words
def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_text)


# 4. remove numbers
def remove_numbers(text):
    return re.sub(r"\d+", "", text)


# 5. remove short words
def remove_short_words(text):
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if len(w) > 2]
    return " ".join(filtered_text)


# 6. lemmatize
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(w) for w in word_tokens]
    return " ".join(lemmatized_text)


# 8. remove non-ascii characters
def remove_non_ascii(text):
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )


# 9. remove extra spaces
def remove_extra_spaces(text):
    return re.sub(" +", " ", text)


# 16. remove extra non-breaking spaces
def remove_extra_non_breaking_spaces(text):
    return re.sub("\xa0+", "", text)


if __name__ == "__main__":
    pass
