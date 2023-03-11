import nltk
import numpy as np
import re
import pickle
import csv
import string
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

lem = WordNetLemmatizer()
nltk.download('stopwords')


def open_tsv(file_name):
    '''open tsv files into lines'''
    tsv_file = open(file_name, errors="ignore")
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    file = []
    for row in read_tsv:
        file.append(row)

    tsv_file.close()
    return file

def pre_process(txt):
    '''Basic text preprocessing, including 
    to lowercases, remove punctuations/odd characters/slash_n/meaningless words, tokenize, lemmatization'''
    # convert to lowercases
    txt = txt.lower()

    # replace punctuations with spaces
    for punc in string.punctuation:
        txt = txt.replace(punc, ' ')

    # remove odd characters (keep alphabets only)
    txt = re.sub(r'[^a-z ]', '', txt)

    # tokenize the txt
    txt = word_tokenize(txt)

    # lemmatization
    txt = [lem.lemmatize(word) for word in txt]

    # stop word removel, too-short word removel
    stop_words = stopwords.words('english')
    txt = [w for w in txt if w not in stop_words and len(w) > 1]

    return txt

def get_inv_idx(raw_txt):
    '''calculate the inverted index and store with nested dictionaries:
    a dictionary stores all the terms
    each term contains a dictionary, stores all related doc and occurance
    '''
    print('Start calculating inverted index...')
    inverted_idx = {}

    # variables keep track of already processed passages
    exist_psg = {}
    count_psg = 0

    for i in range(len(raw_txt)):
        pid =  raw_txt[i][1]

        if pid not in exist_psg:

            exist_psg[pid] = 1
            count_psg += 1

            passage = pre_process(raw_txt[i][3])

            for w in passage:
                if w in inverted_idx:
                    if pid in inverted_idx[w]:
                        inverted_idx[w][pid] += 1
                    else:
                        inverted_idx[w][pid] = 1
                else:
                    inverted_idx[w] = {}
                    inverted_idx[w][pid] = 1
    
    print('Total processed passages:', count_psg)
    print('Finish calculating inverted index')
    return inverted_idx

def save_inv_idx(file, filename):
    '''save inverted index to pickle file'''
    pkl_file = open(filename, "wb")
    pickle.dump(file, pkl_file)
    pkl_file.close()
    print('inverted index saved!')

if __name__ == '__main__':
    print('start cw task 2...')
    ## read the tsv file
    top1000 = open_tsv("candidate-passages-top1000.tsv")


    ## get the inverted index
    inverted_index = get_inv_idx(top1000)


    # store the inverted index
    save_inv_idx(inverted_index, "inverted_index.pkl")

