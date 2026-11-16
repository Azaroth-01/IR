pip install nltk

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt_tab')

# 2. View available English stop words
print("Stop words sample:")
print(stopwords.words('english')[:10])

data = "My name is Shantanu Gaikwad and studying AI&DS Engineering at ADYPSOE."
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(data)

print(f"\nOriginal Word Tokens:\n{word_tokens}")

# Case-insensitive stop word removal
filtered_sentence = []
for w in word_tokens:
    if w.lower() not in stop_words:
        filtered_sentence.append(w)

print(f"\nCase-insensitive stop word removal:\n{filtered_sentence}")

# Case-sensitive stop word removal
filtered_sentence_case = []
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence_case.append(w)

print(f"\nCase-sensitive stop word removal:\n{filtered_sentence_case}")

# Stemming the filtered sentence
ps = PorterStemmer()
stemmed_sentence = [ps.stem(w) for w in filtered_sentence]

print(f"\nStemmed sentence:\n{stemmed_sentence}")
