"""
Contains code for processing a line
"""
import re
import string

import nltk
from keras.preprocessing import sequence
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

from embedding import load_embeddings


class LineProcessor(object):
    def __init__(self):

        self.word2index, self.embedding_matrix = load_embeddings()
        self.ignore_words = ['of', 'and', 'to', 'a']
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = stopwords.words('english')

    def fit(self, line, sentence_length=20):
        tokens = nltk.word_tokenize(line)

        """
        WORD -> INDEX
        """
        new_line = []
        for token in tokens:
            index = self.process_word(token)
            if index is not False:
                new_line.append(index)

        """
            PADDING SENTENCE
        """
        new_padded_line = \
            sequence.pad_sequences([new_line], maxlen=sentence_length, truncating="post", padding="post", value=-1)[0]
        return new_padded_line

    def process_word(self, word):
        """
        Get index of the word as per the word2index in embedding.py
        :param word: str
        :return: index of the word
        """

        word = re.sub(r'[^\w\s]', '', word)
        word = word.lower().strip()

        if word.isdigit() or word in string.punctuation or word in self.stopwords or word in self.ignore_words:
            return False

        try:
            index = self.word2index[word]
        except KeyError:
            lemmatized_word = self.lemmatizer.lemmatize(word)

            try:
                index = self.word2index[lemmatized_word]
            except KeyError:
                return False

        return index
