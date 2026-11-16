import string

document1 = "The quick brown fox jumped over the lazy dog." 
document2 = "The lazy dog slept in the sun."

# Function to clean punctuation and tokenize
def preprocess(doc):
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    cleaned = doc.lower().translate(translator)
    return cleaned.split()

tokens1 = preprocess(document1)
tokens2 = preprocess(document2)

print(f"Token1 : {tokens1}")
print(f"Token2 : {tokens2}\n")

# Combine the tokens into a list of unique terms
terms = list(set(tokens1 + tokens2))
print(f"Terms : {terms}\n")

# Create an empty dictionary to store the inverted index 
inverted_index = {}

# For each term, find the documents that contain it 
for term in terms:
    documents = []
    if term in tokens1: 
        documents.append("Document 1")
    if term in tokens2: 
        documents.append("Document 2")
    inverted_index[term] = documents

# Print inverted index
print("Inverted Index Dictionary:")
for term, documents in inverted_index.items(): 
    print(f"{term} -> {', '.join(documents)}")
