import conllu


def get_sentences(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = [sent for sent in conllu.parse_incr(f)]
    return data


def get_vocabs(data, vocab_list):
    for sentence in data:
        for word in sentence:
            vocab_list.append(word['form'])
    return vocab_list


def get_word_embeddings(file_path):
    word_embeddings = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            word, emb = line.split(maxsplit=1)
            emb = np.fromstring(emb, "f", sep=" ")
            word_embeddings[word] = emb

    return word_embeddings