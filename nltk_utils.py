import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np


stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word): 
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence , all_words ):

    ignore_words = [ '/' , '?' , '.' , ',' , '!' ]

    tokenized_sentence = [stem(w) for w in tokenized_sentence if w not in ignore_words]

    bag = np.zeros(len(all_words) , dtype=np.float32)

    for index , word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1
        else:
            bag[index] = 0
    
    return bag

# a = "How long does shipping take?"
# print(a)
# a = tokenize(a)
# print(a)

# words = ["Organize" , "Organizes" , "organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)

# sentence = ["hello","how","are","you"]
# words = ["hi","hello","I","you","bye","thank","cool"]

# print( bag_of_words(sentence, words))

