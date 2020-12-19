#coding=utf-8
import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.corpora import Dictionary
from gensim.models.wrappers import LdaMallet
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
from gensim.models.phrases import Phrases, Phraser


def preprocessing_and_training(docs, model=None, num_topics=None, random_state=None, download=True):
    # Download and import stopwords library
    if download:
        nltk.download('stopwords')
    # Download and import stopwords library
    stopwds = stopwords.words('english')

    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = str(docs[idx]).lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove stopwords.
    docs = [[token for token in doc if token not in stopwds] for doc in docs]

    # Porter stemmer.
    stemmer = PorterStemmer()
    docs = [[stemmer.stem(token) for token in doc] for doc in docs]

    # Compute bigrams.
    phrases = Phrases(docs, min_count=1, threshold=2)
    bigram = Phraser(phrases)
    for idx in range(len(docs)):
        docs[idx] = bigram[docs[idx]]

    # text processing is done
    query = docs[-1]
    docs = docs[:-1]

    # Remove rare and common tokens.
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    if not model:
        # Make a index to word dictionary.
        temp = dictionary[0]  # This is only to "load" the dictionary.
        id2word = dictionary.id2token

        # Train LDA model.
        path_to_mallet_binary = os.path.join(os.getcwd(), "..", "notebook", "mallet-2.0.8", "bin", "mallet")
        mallet_model = LdaMallet(
            path_to_mallet_binary,
            corpus=corpus,
            id2word=id2word,
            num_topics=num_topics,
            random_seed=random_state
        )
        model = malletmodel2ldamodel(mallet_model)

    return model, corpus, dictionary, docs, query


def LDA_ranking(model, corpus, dictionary, query, N=50):
    # QUERY ORIGINAL #######

    # query topics sorted by score
    initial_query_topics = model.get_document_topics(dictionary.doc2bow(query))
    query_topics = list(sorted(initial_query_topics, key=lambda x: x[1], reverse=True))

    # Get documents with highest probability for each topic. Source: https://stackoverflow.com/a/56814624

    # matrix with the query topics as columns and docs as rows, value is the proba
    topic_matrix = pd.DataFrame(columns=[x[0] for x in query_topics])

    # Loop over all the documents to group the probability of each topic
    for docID in range(len(corpus)):
        topic_matrix.loc[len(topic_matrix)] = 0  # fill with zeros
        topic_vector = OrderedDict(model[corpus[docID]])  # convert list of tuples to OrderedDict to go faster
        for topicID, proba in query_topics:  # only the query topics are relevant
            topic_matrix.at[docID, topicID] = proba * topic_vector.get(topicID, 0)

    # sum probas
    # ids dos docs
    docs = np.array((range(len(corpus))))
    # soma de todas as colunas p/ cada documento
    topic_probas = topic_matrix.sum(axis=1)
    # sorteia da maior p/ a menor proba
    ids = docs[np.argsort(topic_probas[docs])[::-1]]
    # top_docs = topic_probas[ids]

    # proba da palavra w * (proba da palavra 1 da query * proba da palavra 2 da query...)
    # ou obter todas as palavras que a relevância seja > 0
    query_per_word_topics = model.get_document_topics(dictionary.doc2bow(query), per_word_topics=True)
    query_per_word_topics = query_per_word_topics[2]
    query_per_word_topics = [OrderedDict(probas) for word_id, probas in query_per_word_topics]
    doc_ids_to_extend = ids[:N]
    new_words_ids = []

    # Loop over all the documents to get the new query words
    for docID in doc_ids_to_extend:
        per_word_topics = model.get_document_topics(corpus[docID], per_word_topics=True)
        per_word_topics = per_word_topics[2]

        for wordID, topics_probas in per_word_topics:
            for topicID, proba in topics_probas:
                result = proba
                for d in query_per_word_topics:
                    result *= d.get(topicID, 0)

                if result > 0:
                    new_words_ids.append(wordID)

    new_words_ids = list(set(new_words_ids))
    new_query = query + [dictionary[wordID] for wordID in new_words_ids]

    # QUERY EXTENDIDA #######

    # query topics sorted by score
    initial_query_topics = model.get_document_topics(dictionary.doc2bow(new_query))
    query_topics = list(sorted(initial_query_topics, key=lambda x: x[1], reverse=True))

    # Get documents with highest probability for each topic. Source: https://stackoverflow.com/a/56814624

    # matrix with the query topics as columns and docs as rows, value is the proba
    topic_matrix = pd.DataFrame(columns=[x[0] for x in query_topics])

    # Loop over all the documents to group the probability of each topic
    for docID in range(len(corpus)):
        topic_matrix.loc[len(topic_matrix)] = 0  # fill with zeros
        topic_vector = OrderedDict(model[corpus[docID]])  # convert list of tuples to OrderedDict to go faster
        for topicID, proba in query_topics:  # only the query topics are relevant
            topic_matrix.at[docID, topicID] = proba * topic_vector.get(topicID, 0)

    # sum probas
    return np.array(topic_matrix.sum(axis=1))
