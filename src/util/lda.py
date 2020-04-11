#coding=utf-8
from collections import OrderedDict
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.phrases import Phrases, Phraser


def get_corpus_for_docs(docs):
    """
    Faz o tratamento da base e retorna o corpus, dicionário e query tratada
    """
    # Download and import stopwords library
    nltk.download('stopwords')
    nltk.download('wordnet')
    stopwds = stopwords.words('english')

    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = str(docs[idx]).lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    # Remove stopwords.
    #stopwds.add('new word')
    docs = [[token for token in doc if token not in stopwds] for doc in docs]

    # Lemmatize the documents.
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    # Compute bigrams.
    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    phrases = Phrases(docs, min_count=20)
    bigram = Phraser(phrases)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)

    # text processing is done
    query = docs[-1]
    docs = docs[:-1]

    # Remove rare and common tokens.
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    return corpus, dictionary, query


def get_top_documents(corpus, dictionary, query, num_topics, random_state):
    """
    Retorna TOP documentos da query.

    Recebe corpus, dicionário, query de busca, número de
    tópicos e a semente do random.
    """
    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    # Train LDA model.
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        alpha='auto',
        eta='auto',
        gamma_threshold=0.01,  # se o resultado mudar menos de 1% as iterações são interrompidas
        num_topics=num_topics,
        random_state=random_state
    )

    # query topics sorted by score
    # query topics sorted by score
    initial_query_topics = model.get_document_topics(dictionary.doc2bow(query))
    query_topics = list(sorted(initial_query_topics, key=lambda x: x[1], reverse=True))

    # Get documents with highest probability for each topic. Source: https://stackoverflow.com/a/56814624

    # matrix with the query topics as columns and docs as rows, value is the proba
    topic_ids = [x[0] for x in query_topics]
    topic_matrix = pd.DataFrame(columns=topic_ids)

    # Loop over all the documents to group the probability of each topic
    for docID in range(len(corpus)):
        topic_matrix.loc[len(topic_matrix)] = 0  # fill with zeros
        topic_vector = OrderedDict(model[corpus[docID]])  # convert list of tuples to OrderedDict to go faster
        for topicID in topic_ids:  # only the query topics are relevant
            topic_matrix.at[docID, topicID] = topic_vector.get(topicID, 0)

    # sum probas
    return np.array(topic_matrix.sum(axis=1))
