import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import pandas as pd
import timeit
import pickle
np.random.seed(229)

EMBEDDING_FILENAME = 'lda_embeddings'

"""
LDA processing. Adapted from https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
"""

stemmer = SnowballStemmer('english')

def lemmatize_stemming(caption):
    """Get stemmings of words (normalize words)"""
    return stemmer.stem(WordNetLemmatizer().lemmatize(caption, pos='v'))


def preprocess(caption):
    """Remove stop words and lemmatize words"""
    result = []
    for token in gensim.utils.simple_preprocess(caption):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def create_bow_corpus(user_captions_df, no_below=15, no_above=0.5, keep_n=10000):
    """Create bag of words corpus from profile data frame"""
    processed_captions = user_captions_df['description'].map(preprocess)
    dictionary = gensim.corpora.Dictionary(processed_captions)
    # Remove items that appear fewer than
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    bow_corpus = [dictionary.doc2bow(user_doc) for user_doc in processed_captions]
    return bow_corpus, dictionary


def create_tfidf_corpus(bow_corpus):
    """Compute tfidf scores for each bow document in bow corpus"""
    tfidf = models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf[bow_corpus]
    return tfidf_corpus


def train_lda_model(corpus, dictionary, num_topics=50):
    """Construct, train and return lda model from corpus (either tfidf or bow corous)
    and dictionary.
    """
    lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=4)
    return lda_model


def construct_lda_embedding(lda_model, processed_user_captions):
    """Create lda embedding for each user using a trained lda model and the user's
    captions. Note that the processed_user_captions should be bow or tfidf processed,
    depending on how the lda model was trained.
    """
    # print(processed_user_captions)
    score_tuples = lda_model.get_document_topics(processed_user_captions, minimum_probability=0)
    # print(score_tuples)
    scores = list(map(lambda t: t[-1], score_tuples))
    # print(scores)
    return np.array(scores)


def main():
    """Main running script"""
    model_path = 'lda.model'
    save_model = False
    # Load csv
    data_path = 'data/captions_dataset.csv'
    df = pd.read_csv(data_path, header=0)
    # Isolate caption column
    # Create bow corpus
    bow_corpus, dictionary = create_bow_corpus(df)
    # print(bow_corpus[4])
    # print(dictionary[1])
    # Create tfidf corpus
    tfidf_corpus = create_tfidf_corpus(bow_corpus)
    # print(tfidf_corpus[4])
    # Train and save lda modellda model
    if save_model:
        lda_model = train_lda_model(tfidf_corpus, dictionary)
        lda_model.save(model_path)
    else:
        lda_model = gensim.models.ldamodel.LdaModel.load(model_path)
    # Print topics
    # print(lda_model.show_topics(num_topics=50, num_words=1))

    # Create lda profile embeddings
    usernames = df['alias'].unique().tolist()
    embeddings_dict = dict()
    for idx, username in enumerate(usernames):
        print("Processing user: {}, {}".format(idx, username))
        processed_user_captions = tfidf_corpus[idx]
        # Ignore user if no text
        if len(processed_user_captions) < 1:
            continue
        # Generate embedding
        # start = timeit.default_timer()
        embedding = construct_lda_embedding(lda_model, processed_user_captions)
        # elapsed = timeit.default_timer() - start
        # print("Generating embedding took {}s".format(elapsed))
        # print(embedding)
        # Store embedding
        embeddings_dict[username] = embedding
        # Save embeddings every 80 users
        if idx % 80 == 0:
            with open(EMBEDDING_FILENAME+'.pkl', 'wb') as f:
                    pickle.dump(embeddings_dict, f)
    # Save final embeddings
    with open(EMBEDDING_FILENAME+'.pkl', 'wb') as f:
            pickle.dump(embeddings_dict, f)
    # Combine usernames and embeddings
    # save embeddings

if __name__ == '__main__':
    main()
