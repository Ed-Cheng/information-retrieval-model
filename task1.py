import nltk
import numpy as np
import re
import matplotlib.pyplot as plt
import string
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()


def open_txt(file_name):
    '''open txt files into one string'''
    f = open(file_name, "r", errors="ignore")
    file = f.read()
    f.close()
    return file

def pre_process(txt):
    '''Basic text preprocessing, including 
    to lowercases, remove punctuations/odd characters/slash_n, tokenize, lemmatization'''
    # convert to lowercases
    txt = txt.lower()

    # remove \n
    txt = txt.replace('\n', ' ')

    # replace punctuations with spaces
    for punc in string.punctuation:
        txt = txt.replace(punc, ' ')

    # remove odd characters (keep alphabets only)
    txt = re.sub(r'[^a-z ]', '', txt)

    # tokenize the txt
    txt = word_tokenize(txt)

    # lemmatization
    txt = [lem.lemmatize(word) for word in txt]

    return txt

def save2dic(words):
    '''store lemmatized words to a dictionary and count word frequncy'''
    store = {}
    for i in range(len(words)):
        if words[i] in store:
            store[words[i]] += 1
        else: 
            store[words[i]] = 1

    return store


if __name__ == '__main__':
    print('start cw task 1...')
    ## read the txt file
    passage_file = open_txt("passage-collection.txt")


    ## pre-process the txt
    passage_processed = pre_process(passage_file)


    ## store the processed txt into dictionary
    term_dictionary = save2dic(passage_processed)
    term_size = len(term_dictionary)
    print("size of the identified index of terms (vocabulary):", term_size)


    ## visualization
    visual = np.ones((term_size, 2))
    for count, i in enumerate(term_dictionary):
        visual[count, 0] = term_dictionary[i]

    # sort by first column (occurance of words)
    order = visual[visual[:, 0].argsort()]
    order = order[::-1]

    # normalize the frequency
    all_terms = sum(order[:, 0])
    for i in range(term_size):
        order[i, 1] = order[i, 0]/all_terms

    # Zipfian numerator
    numerator = sum([i**-1 for i in range(1, term_size+1)])

    # plot our result and Zipf's law onto the same figure 
    x = [i for i in range(1, term_size+1)]
    y = order[:, 1]
    Zipfian = [(1/(i*numerator)) for i in range(1, term_size+1)]

    plt.plot(x,y, color='r', label='data')
    plt.plot(x,Zipfian, color='g', label="Zipf's law")
    plt.xlabel('frequency ranking (log)')
    plt.ylabel('normalised frequency (log)')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('ZipLaw.png')
    