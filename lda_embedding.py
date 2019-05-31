import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(229)

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
    bow_corpus = [dictionary.doc2bow(user_doc) for user_doc in processed_docs]
    return bow_corpus, dictionary


def create_tfidf_corpus(bow_corpus):
    """Compute tfidf scores for each bow document in bow corpus"""
    tfidf = models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf[bow_corpus]
    return tfidf_corpus


def train_lda_model(corpus, dictionary, num_topics=20):
    """Construct, train and return lda model from corpus (either tfidf or bow corous)
    and dictionary.
    """
    lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=2, workers=4)
    return lda_model


def construct_lda_embeddings(lda_model, num_topics, processed_user_captions):
    """Create lda embedding for each user using a trained lda model and the user's
    captions. Note that the processed_user_captions should be bow or tfidf processed,
    depending on how the lda model was trained.
    """
    N = len(processed_user_captions)
    lda_embeddings = np.zeros((N, num_topics))
    for i, captions in enumerate(processed_user_captions):
        score_tuples = lda_model[captions]
        scores = list(map(lambda t: t[-1], score_tuples))
        lda_embeddings[i] = np.array(scores)
    return lda_embeddings


def main():
    """Main running script"""
    # Load csv
    # Isolate caption column
    # Create bow corpus
    # Create tfidf corpus
    # Train lda model
    # Print topics
    # Save lda model
    # Create lda profile embeddings
    # Combine usernames and embeddings
    # save embeddings
