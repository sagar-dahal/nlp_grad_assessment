import conllu
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.utils import to_categorical

def get_sentences(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = [sent for sent in conllu.parse_incr(f)]
    return data


def get_vocabs(data, vocab_list):
    for sentence in data:
        for word in sentence:
            vocab_list.append(word['form'])
    return vocab_list


def glove_embeddings(file_path):
    word_embeddings = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            word, emb = line.split(maxsplit=1)
            emb = np.fromstring(emb, "f", sep=" ")
            word_embeddings[word] = emb

    return word_embeddings


def get_missed_vocabs(vocabs, word_emb):
    missed_vocabs = []
    for vocab in vocabs:
        if vocab.lower() not in word_emb:
            missed_vocabs.append(vocab)

    return list(set(missed_vocabs))


def remove_sent_missed_vocab(data, missed_vocabs):
    for index, sentence in enumerate(data):
        for word in sentence:
            if word['form'] in missed_vocabs:
                del data[index]
                break

    return data


def pos_embeddings(size):
    upos = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 
            'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 
            'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

    upos_list = np.array([ x for x in range(1, len(upos)+1)])
    return to_categorical(upos_list-1, num_classes=size)


def dep_embeddings(size):
    dep = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case',
            'cc', 'ccomp', 'clf', 'compound', 'conj', 'cop',
            'csubj', 'dep', 'det', 'discourse', 'dislocated', 'expl',
            'fixed', 'flat', 'goeswith', 'iobj', 'list', 'mark',
            'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan',
            'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp']

    dep_list = np.array([ x for x in range(1, len(dep)+1)])
    return to_categorical(dep_list-1, num_classes=size)