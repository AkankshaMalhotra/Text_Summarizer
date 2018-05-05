import _pickle as cPickle


def load():
    with open("model/term_document_matrix.pkl","rb") as g:
        term_document_matrix = cPickle.load(g)
    term_document_matrix_inv = term_document_matrix.T
    with open("model/term_document_matrix_index.pkl", "rb") as g:
        index_tdm = cPickle.load(g)
    return term_document_matrix_inv, index_tdm
