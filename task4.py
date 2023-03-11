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


def open_pkl(file_name):
    '''open pickle files that stored nested dictionary'''
    pkl_file = open(file_name, "rb")
    file = pickle.load(pkl_file)
    pkl_file.close()

    return file

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

def laplace(pre_query, pre_psg, D, V):
    '''performs laplace estimate for one passage'''
    prob = 1
    for term in np.unique(pre_query):
        occur = pre_psg.count(term)
        prob *= ((occur+1) / (D+V))

    return np.log(prob)

def lidstone(pre_query, pre_psg, epsilon, D, V):
    '''performs lidstone correction for one passage'''
    prob = 1
    for term in np.unique(pre_query):
        occur = pre_psg.count(term)
        prob *= ((occur+epsilon) / (D + epsilon*V))

    return np.log(prob)

def dirichlet(pre_query, pre_psg, D, full_collection, mu):
    '''performs lidstone correction for one passage'''
    like = 0
    for term in np.unique(pre_query):
        if term in inv_idx:
            fqi = pre_psg.count(term)
            cqi = sum([inv_idx[term][i] for i in inv_idx[term]])

        like += np.log((D/(D+mu))*(fqi/D) + (mu/(D+mu))*(cqi/full_collection))

    return like


def language_model(top1000, test_queries, inv_idx, method):
    '''Implement query likelihood models, either laplace, lidstone, or dirichlet'''
    print(f'Start calculating with {method} ...')

    ## initialize parameters
    # number of unique words in the entire collection (vocabulary size)
    V = len(inv_idx)
    epsilon = 0.1
    mu = 50

    # empty list to store all scores
    score_all = []

    for i in range(len(test_queries)):

        pre_query = pre_process(test_queries[i][1])

        # empty list to store top 100 scores
        score_top100 = []
        for p in top1000:
            pid = int(p[1])
            qid = int(p[0])
            if qid == int(test_queries[i][0]):
                pre_psg = pre_process(p[3])

                # number of words in the passage
                D = len(pre_psg)

                if method == 'laplace':
                    score = laplace(pre_query, pre_psg, D, V)
                elif method == 'lidstone':
                    score = lidstone(pre_query, pre_psg, epsilon, D, V)
                elif method == 'dirichlet':
                    full_collection = mu * len(top1000)
                    score = dirichlet(pre_query, pre_psg, D, full_collection, mu)

                score_top100.append([score, pid])
        
        score_top100.sort(reverse=True)
        score_all.append(score_top100[:100])
    
    print(f'End calculating with {method} ...')
    return score_all

def save2csv(file_name, file):
    '''save to the csv file required'''
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        for q in range(len(file)):
            for top in range(len(file[q])):
                row = file[q][top]

                # save in the format: qid,pid,score
                config = [test_queries[q][0],row[1],row[0]]
                writer.writerow(config)
    print(f'----- {file_name} saved -----')
    f.close


if __name__ == '__main__':
    print('start cw task 4...')
    ## read tsv/pickle file and append passsage word count at the end of top1000
    inv_idx = open_pkl("inverted_index.pkl")
    top1000 = open_tsv("candidate-passages-top1000.tsv")
    test_queries = open_tsv("test-queries.tsv")

    ## Implement laplace, lidstone, and dirichlet
    laplace_outcome = language_model(top1000, test_queries, inv_idx, 'laplace')
    save2csv('laplace.csv', laplace_outcome)

    lidstone_outcome = language_model(top1000, test_queries, inv_idx, 'lidstone')
    save2csv('lidstone.csv', lidstone_outcome)

    dirichlet_outcome = language_model(top1000, test_queries, inv_idx, 'dirichlet')
    save2csv('dirichlet.csv', dirichlet_outcome)
