import _pickle as cPickle
import os
import string
from itertools import chain

from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from config import data_folder


def read_raw_docs():
    '''
    Reading bbc news text from various folders
    :return: list of news documents
    '''
    ar = os.listdir(data_folder)
    news_doc = []
    for news_type in ar:
        path = data_folder+"/"+news_type
        try:
            for article in os.listdir(path):
                with open(path+"/"+article, "r") as g:
                    news_doc.append(g.read())
        except:
            continue
    print(len(news_doc))
    return news_doc


def get_wordnet_pos(treebank_tag):
    '''
    Gets pos tags as defined in penn treebank corpus and returns wordnet pos tags
    :param treebank_tag: Penn Treebank POS tags
    :return: Wordnet pos tags if present else returns an empty character
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def get_lemmatized_documents(news_doc):
    '''
    Iterates through list of news documents. Does the sentence tokenization of each document. Further it finds pos_tags
    for each word after doing word tokenization. Converts penn postags to wordnet pos tags using the function get_
    wordnet_pos(). We use wordnet lemmatizer with word and its pos tag as arguments. Finally we join the sentences where
    the words are changed to its corresponding lemma.
    :param news_doc: list of news documents
    :return: Lemmatized list of news documents
    '''
    lemmatizer = WordNetLemmatizer()
    new_doc = []
    for each_doc in tqdm(news_doc):
        doc = []
        for sent in sent_tokenize(each_doc):
            tags = pos_tag(word_tokenize(sent))
            doc.extend(list(map(
                lambda i: lemmatizer.lemmatize(i[0], get_wordnet_pos(i[1])) if get_wordnet_pos(i[1]) != "" else i[0],
                tags)))
        new_doc.append(" ".join(doc))
    return new_doc


def preprocessing_text(new_doc):
    '''

    :param new_doc:
    :return:
    '''
    punct = string.punctuation
    punct = punct.replace("-", "")
    translator = str.maketrans("\n-", '  ', punct + "£" + string.digits)
    punc_rem = [doc.translate(translator).lower() for doc in new_doc]
    return punc_rem


def train(news_doc):
    stopword = set(stopwords.words('english'))
    vocabulary = list(map(lambda doc: list(filter(lambda x: x != '', word_tokenize(doc))), news_doc))
    vocabulary = set(chain(*vocabulary))
    vectorizer = CountVectorizer(vocabulary=vocabulary, stop_words=stopword)
    term_document_matrix = vectorizer.fit_transform(news_doc)
    index_tdm = vectorizer.vocabulary_
    return term_document_matrix, index_tdm


def main():
    news_docs = read_raw_docs()
    lemmatized_docs = get_lemmatized_documents(news_docs)
    preprocessed_docs = preprocessing_text(lemmatized_docs)
    tdm,i_tdm = train(preprocessed_docs)
    with open("model/term_document_matrix.pkl", "wb") as g:
        cPickle.dump(tdm, g)
    with open("model/term_document_matrix_index.pkl", "wb") as g:
        cPickle.dump(i_tdm, g)

main()
