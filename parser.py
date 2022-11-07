from files import file_paths
import utilities

# size of vector embeddings
size = 50

# loading the dataset
data_train = utilities.get_sentences(files['eng-train'])
data_dev = utilities.get_sentences(files['eng-dev'])
data_test = utilities.get_sentences(files['eng-test'])

# getting list of unique vocabulary in the datasets
vocab = []

vocab = utilities.get_vocabs(data_train, vocab)
vocab = utilities.get_vocabs(data_dev, vocab)
vocab = utilities.get_vocabs(data_test, vocab)

vocab = set(vocab)

# loading GloVe embedding
word_emb = utilities.glove_embeddings(file_paths['glove-'+str(size)])

# words in dataset that are not present in word embedding
missed_vocab = utilities.get_missed_vocabs(vocab, word_emb)

# Removing the sentences in the datasets that have missed vocabulary
data_train_trim = utilities.remove_sent_missed_vocab(data_train, missed_vocab)
data_dev_trim = utilities.remove_sent_missed_vocab(data_dev, missed_vocab)
data_test_trim = utilities.remove_sent_missed_vocab(data_test, missed_vocab)


# Getting vocabulary from the trimmed dataset
vocab_trim = []

vocab_trim = utilities.get_vocabs(data_train_trim, vocab_trim)
vocab_trim = utilities.get_vocabs(data_dev_trim, vocab_trim)
vocab_trim = utilities.get_vocabs(data_test_trim, vocab_trim)

vocab_trim = set(vocab_trim)

# Getting embeddings for POS and DEPS
pos_emb_mat = utilities.pos_embeddings(size)

# Get dependencies embeddings
deps_emb_mat = utilities.deps_embeddings(size)