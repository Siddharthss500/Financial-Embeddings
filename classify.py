# Import necessary libraries
import pandas as pd
import re
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import gensim
import gensim.downloader
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Bidirectional
import keras.backend as K


class DataFrame():

    def __init__(self, path):
        self.path = path
        self.df = None
        self.X = None
        self.Y = None

    def _append_df(self):
        # self.df = pd.read_csv(self.path+"Sentences_AllAgree.txt", sep=".@", names=['text', 'label'], engine='python')
        all_files = [file for file in os.listdir('data/FinancialPhraseBank-v1.0/') if file.startswith("Sent")]
        for file in all_files:
            df = pd.read_csv(self.path+file, sep=".@", names=['text', 'label'], engine='python')
            if self.df is None:
                self.df = df.copy()
            self.df = self.df.append([self.df, df])

    def _preprocess_text(self, text):
        symbols = re.compile('[^A-Za-z0-9(),!?\'\`]')
        # Remove uneccessary symbols, stop words and convert to lower case
        text = text.replace('\n', ' ').lower()
        text = symbols.sub(' ', text)
        return text

    def _get_train_test(self):
        # Split the data into train/test
        self.X = list()
        sentences = list(self.df["text"])
        for sen in sentences:
            self.X.append(self._preprocess_text(sen))
        labels = self.df['label']
        self.Y = labels.values
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(self.Y)
        encoded_Y = encoder.transform(self.Y)
        # convert integers to dummy variables (i.e. one hot encoded)
        self.Y = np_utils.to_categorical(encoded_Y)

    def create_df(self):
        self._append_df()
        self._get_train_test()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        print(len(X_train))
        print(len(X_test))
        return X_train[0:5000], X_test, y_train[0:5000], y_test


def tokenize_padding(X_train, X_test, maxlen=500):
    # Tokenize
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    vocab_size = len(tokenizer.word_index) + 1
    # Pad sequences
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    return X_train, X_test, vocab_size, tokenizer


def create_embed(tokenizer, vocab_size, pretrained):
    words_found = 0
    # To load pre-trained model
    w2v = gensim.downloader.load(pretrained)
    # To load a model trained on our data
    # w2v = gensim.models.Word2Vec.load("word2vec.model")
    embedding_matrix = np.zeros((vocab_size, w2v.vector_size))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, index in tokenizer.word_index.items():
        if word in w2v:
            embedding_vector = w2v[word]
            if embedding_vector is not None:
                words_found += 1
                embedding_matrix[index] = embedding_vector
        else:
            embedding_matrix[index] = np.random.uniform(-0.25, 0.25, 300)
    print("Total number of words are ", vocab_size)
    print("Total number of words found in pre-trained embeddings are ", words_found)
    return embedding_matrix


# Note : This function was taken from medium : https://medium.com/@aakashgoel12/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d
def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def DNN_model(embed_size, embedding_matrix, vocab_size, maxlen):
    deep_inputs = Input(shape=(maxlen,))
    embedding_layer = Embedding(vocab_size, embed_size, weights=[embedding_matrix], trainable=False)(deep_inputs)
    # BiLSTM_Layer_1 = LSTM(128)(embedding_layer)
    BiLSTM_Layer_1 = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
    BiLSTM_Layer_2 = Bidirectional(LSTM(64))(BiLSTM_Layer_1)
    dense_layer_1 = Dense(3, activation='softmax')(BiLSTM_Layer_2)
    model = Model(inputs=deep_inputs, outputs=dense_layer_1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',  get_f1])
    return model


def main():
    my_df = DataFrame('data/FinancialPhraseBank-v1.0/')
    X_train, X_test, y_train, y_test = my_df.create_df()
    print("Loaded dataframe")
    X_train, X_test, vocab_size, tokenizer = tokenize_padding(X_train, X_test, 500)
    print("Tokenized the sentences")
    # Choose the pretrained models - word2vec-google-news-300, glove-wiki-gigaword-300, fasttext-wiki-news-subwords-300
    embed_mat = create_embed(tokenizer, vocab_size, 'word2vec-google-news-300')
    print("Created the embeddings for each token in the sentence")
    my_LSTM_model = DNN_model(300, embed_mat, vocab_size, 500)
    print(my_LSTM_model.summary())
    my_LSTM_model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.2)
    print("Model trained")
    LSTM_score = my_LSTM_model.evaluate(X_test, y_test, verbose=1)
    print("Model score : ", LSTM_score)
    return


if __name__ == "__main__":
    main()
