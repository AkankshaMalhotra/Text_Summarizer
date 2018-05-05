from __future__ import division

import math
import string

import numpy as np
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet

from config import lemmatizer


class TextSummarizer:
    def __init__(self, input_text, informative, term_document_matrix_inv, index_tdm):
        """
        :param input_text: text given by the user
        :param informative: flag to tell whether user wants indicative(0) or informative(1)
        :param term_document_matrix_inv: term document matrix
        :param index_tdm: index of term document matrix
        """
        self.text = input_text
        self.term_document_matrix_inv = term_document_matrix_inv
        self.index_tdm = index_tdm
        self.informative = informative

    def get_flag(self):
        """
        :return: flag to tell whether user wants indicative(0) or informative(1)
        """
        return self.informative

    def get_input(self):
        """
        :return: input text
        """
        return self.text

    def get_tdm_row(self, term):
        """
        :param term: word to find document frequencies
        :return: row of document frequencies for the word
        """
        return self.term_document_matrix_inv[self.index_tdm[term]]

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """
        Gets pos tags as defined in penn treebank corpus and returns wordnet pos tags
        :param treebank_tag: Penn Treebank POS tags
        :return: Wordnet pos tags if present else returns an empty character
        """
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

    def get_score(self, sent, word_title):
        """
        Get the scores of different sentences based on the title, tf-idf scores obtained through training it on 2000
        news articles. The input is first preprocessed.
        :param sent: list of sentences to be score
        :param word_title: list of words in the title
        :return: descending order of indices of the list sent based on their score
        """
        score = []
        for each_sent in sent:
            s = 0
            s = list(map(lambda x: s + 1 if x in each_sent else s + 0, word_title))
            s = np.sum(s)
            punctuations = string.punctuation
            punctuations = punctuations.replace("-", "")
            translator = str.maketrans("\n-", '  ', punctuations + "£" + string.digits)
            each_sent = each_sent.translate(translator).lower()
            stop_word = set(stopwords.words('english'))
            wor = [i for i in word_tokenize(each_sent) if i not in stop_word]
            words = [
                lemmatizer.lemmatize(i.lower(), self.get_wordnet_pos(pos_tag([i.lower()])[0][1])) if self.get_wordnet_pos(
                    pos_tag([i.lower()])[0][1]) != '' else i.lower() for i in wor]
            sc_each_word = 0
            for i in words:
                try:
                    tdm_row = self.get_tdm_row(i)
                    sc_each_word += np.sum(tdm_row) / tdm_row.shape[0]
                except KeyError:
                    continue
            score.append(np.sum(sc_each_word + s))
        score = np.array(score)
        score_max_index = np.argsort(-score)
        return score_max_index

    @property
    def get_summary(self):
        '''
        Split the input text into para. Check if the request is made for indicative summary or informative and proceed
         accordingly. The method for scoring the sentences is same. The difference is in selecting the top sentences. In
         informative we select top k sentences from each para. The function is min(math.ceil(len(sent) / 3), 5). whereas
          in indicative we select top k sentences from whole article. The function is max(math.ceil(len(sent) / 3), 5)
        :return: summary of the article
        '''
        para = self.get_input().split("\r\n\r\n")
        title = para[0]
        word_title = word_tokenize(title)
        if int(self.get_flag()):
            summary_each_para =[]
            for each_para in para:
                sent = sent_tokenize(each_para)
                score_max_index = self.get_score(sent, word_title)
                summary_index = score_max_index[:min(math.ceil(len(sent) / 3), 5)]
                summary_index = sorted(summary_index)
                summary_each_para.append(" ".join([sent[i] for i in summary_index]))
            summary = "\n\n".join(summary_each_para)
            return summary
        else:
            sent = sent_tokenize(self.get_input())
            score_max_index = self.get_score(sent, word_title)
            summary_index = score_max_index[:max(math.ceil(len(sent) / 3), 5)]
            summary_index = sorted(summary_index)
            summary = " ".join([sent[i] for i in summary_index])
            return summary
