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

def cal_cos_sim(a, b):
    '''calculate cosine similarity'''
    sim = (a @ b)/(np.linalg.norm(a) * np.linalg.norm(b))
    return sim

def cal_tf_idf(txt, N, inv_idx):
    ''' calculate the tf-idf of a pre-processed text
     Args:
        txt:        pre-processed text
        N:          number of documents in collection
        inv_idx:    inverted index of terms    
    '''
    term_size = len(inv_idx)
    key_list = list(inv_idx)
    
    tf_idf = np.zeros(term_size)
    for t in np.unique(txt):
        # number of documents in which term t appears
        if t in inv_idx:
            n_t = len(inv_idx[t])
            idf = np.log10(N/n_t)
            tf = txt.count(t)

            term_idx = key_list.index(t)
            tf_idf[term_idx] = tf * idf

    return tf_idf

def top100tfidf_cos_sim(top1000, test_queries, inv_idx, N):
    '''calculate top 100 cosine similarity per query'''
    print('Start calculating top 100 cosine similarity...')

    # empty list to store all cosine similarity
    cos_sim_all = []

    for i in range(len(test_queries)):
        pre_query = pre_process(test_queries[i][1])
        tf_idf_query = cal_tf_idf(pre_query, N, inv_idx)

        # empty list to store top 100 cosine similarity
        cos_sim_top100 = []
        for p in top1000:
            pid = int(p[1])
            qid = int(p[0])

            # calculate the cosine similarity that match (at most 1000)
            if qid == int(test_queries[i][0]):

                pre_psg = pre_process(p[3])
                tf_idf_passage = cal_tf_idf(pre_psg, N, inv_idx)

                cosine_similarity = cal_cos_sim(tf_idf_query, tf_idf_passage)
                cos_sim_top100.append([cosine_similarity, pid])
        
        # find the top 100 cosine similarity and store them
        cos_sim_top100.sort(reverse=True)
        cos_sim_all.append(cos_sim_top100[:100])

    print('End calculating top 100 cosine similarity')
    return cos_sim_all

def cal_BM25(q_terms, k1, k2, b, avdl, passage, N):
    '''calculate BM25 score for one passage (pid)'''
    dl = passage[4]
    K = k1 * ((1-b) + b * (dl/avdl))

    pre_psg = pre_process(passage[3])

    BM25 = 0
    for term in np.unique(q_terms):
        if term in inv_idx:
            n = len(inv_idx[term])
            f = pre_psg.count(term)
            qf = q_terms.count(term)

            BM25_1 = np.log(1 / ((n + 0.5)/(N - n + 0.5)))
            BM25_2 = ((k1 + 1) * f) / (K + f)
            BM25_3 = ((k2+1) * qf) / (k2 + qf)

            BM25 += BM25_1 * BM25_2 * BM25_3

    return BM25

def top100BM25(top1000, test_queries, N):
    '''calculate top 100 BM25 score per query'''
    print('Start calculating top 100 BM25...')
    k1 = 1.2
    k2 = 100
    b = 0.75
    avdl = 0
    for i in range(len(top1000)):
        avdl += top1000[i][4]
    avdl /= len(top1000)

    # empty list to store all BM25
    BM25_all = []

    for i in range(len(test_queries)):
        pre_query = pre_process(test_queries[i][1])
        
        # empty list to store top 100 BM25
        BM25_top100 = []
        for p in top1000:
            pid = int(p[1])
            qid = int(p[0])
            if qid == int(test_queries[i][0]):

                BM25_p = cal_BM25(pre_query, k1, k2, b, avdl, p, N)
                BM25_top100.append([BM25_p, pid])
        
        BM25_top100.sort(reverse=True)
        BM25_all.append(BM25_top100[:100])

    print('End calculating top 100 BM25')
    return BM25_all

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
    print('start cw task 3...')
    ## read tsv/pickle file and append passsage word count at the end of top1000
    inv_idx = open_pkl("inverted_index.pkl")
    read_top1000 = open_tsv("candidate-passages-top1000.tsv")
    test_queries = open_tsv("test-queries.tsv")

    top1000 = []
    for row in read_top1000:
        row.append(len(row[3].split()))
        top1000.append(row)


    ## Get the outcomes for csv files (cosine similarity and BM25 score)
    # number of documents in collection
    N = len(top1000)

    # cosine similarity
    cosine_similarity = top100tfidf_cos_sim(top1000, test_queries, inv_idx, N)
    save2csv('tfidf.csv', cosine_similarity)

    # BM25 score
    BM25 = top100BM25(top1000, test_queries, N)
    save2csv('bm25.csv', BM25)
