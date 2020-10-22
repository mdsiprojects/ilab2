"""
Text preprocessing functions.
"""
# %%
def tokenize_sentence(text):
    """
    Wrapper for nltk sent_tokenize.
    Return a sentence-tokenized copy of text, using NLTK’s recommended sentence tokenizer.
    Args:
        text (string): text to split into sentences

    Returns:
        list: list of sentence tokens
    """
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    return(sentences)


# %%
def tokenize_word(text,
                  normalise_case=True,
                  keep_numerics=False,
                  shortwords=3,
                  remove_stopwords=True,
                  lemmatize=True,
                  gaps=False,
                  addl_stopwords=[]):
    """
    Tokenisation function, accepts a text to be tokenised, and applies the following steps according:
    1. normalise case to lower
    2. tokenise using nltk RegexpTokenizer with pattern: r'\w+|$[0-9]+|\S+&[^.<>]'
    3. remove numerics from list of tokens
    4. remove short words
    5. remove english language stop words and custom stop words
    6. lemmatise tokens to base form

    Args:
        text (string): text to split into word tokens
        normalise_case (bool, optional): enable/disable case normalisation. Defaults to True.
        keep_numerics (bool, optional): to keep or remove numerics. Defaults to False.
        shortwords (int, optional): length of short words to be excluded. Defaults to 3.
        remove_stopwords (bool, optional): enable or disable removing of stop words. Defaults to True.
        lemmatize (bool, optional): enable or disable lemmatisation. Defaults to True.
        gaps (bool, optional): True to find the gaps instead of the work tokens. Defaults to False.
        addl_stopwords (list, optional): additional stop words to be excluded from tokenisation. Defaults to [].

    Returns:
        list: list of tokens
    """

    from nltk.tokenize import RegexpTokenizer
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer

    # 1. to lower
    if normalise_case is True:
        text = text.lower()
    # 2. tokenise
    tokenPattern = r'\w+|$[0-9]+|\S+&[^.<>]'
    # potential changes to tokenpatter: (\d+\,?\d+(?|\.\d+)) to
    # capture full numbers with decimal places
    reTokeniser = RegexpTokenizer(tokenPattern, gaps=gaps)
    # , gaps=True -> return gaps list)

    reTokens = reTokeniser.tokenize(text)

    # 3. isnumeric
    if keep_numerics is False:
        reTokens = [token for token in reTokens if not token.isnumeric()]
        # alternatively test for token.isalpha()
        # remove anything that is not aplha

    # 4. remove short short words
    if shortwords > 0:
        reTokens = [token for token in reTokens if len(token) > shortwords]

    # 5. remove stop words

    # auspol_stopwords = ['house', 'senate', 'parliament', 'speaker', 'president', 'member', 'senator', 'government', 'opposition', 'bill', 'debate', 'motion', 'question', 'petition', 'amendment', 'australia', 'australian', 'people', 'minister', 'legislation', 'would', 'think', 'million', 'billion', 'matter', 'issue', 'also', 'many', 'nation', 'national', 'place', 'year', 'time', 'said', 'party']

    if remove_stopwords is True:
        stop_words = set(stopwords.words('english'))
        reTokens = [token for token in reTokens if token not in stop_words and token not in addl_stopwords]

    # 6. lemmatise
    if lemmatize is True:
        lemma = WordNetLemmatizer()
        reTokens = [lemma.lemmatize(token) for token in reTokens]

    return(reTokens)


def get_ngram_module(word_list, prev_ngram=None, min_count=20, threshold=1):
    """
    Create an ngram module, used to generate bi-grams or higher order n-grams

    Args:
        word_list (pandas.Series or list): list of tokens
        prev_ngram (gensim.model.Phrases, optional): a lower order Phrases module, if exists, the function will return 1+ higher order ngram. Defaults to None.
        min_count (int, optional): Ignore all words and bigrams with total collected count lower than this value.. Defaults to 20.
        threshold (int, optional): Score threshold for forming the ngrams (higher means fewer ngrams). Defaults to 1.

    Returns:
        gensim.model.Phrases: n-gram module
    """
    # words_list is a pd.series of word lists
    # or a bigram model to get trigram

    from gensim.models import Phrases
    import pandas as pd

    if isinstance(word_list, pd.Series) is True:
        sents = list(word_list)
    elif isinstance(word_list, list) is True:
        sents = word_list
    else:
        return(None)

    if isinstance(prev_ngram, Phrases):
        sents = prev_ngram[sents]

    ngram = Phrases(sents, min_count=min_count, threshold=threshold)

    return(ngram)

# %%
# make dictionary out of a series of words for documents


def make_dictionary(series_words, large_list=False):
    """
    Make a dictionary out of a series of words for documents
    Args:
        series_words (pandas.Series): documents' tokens, each element is a list of tokens for a document
        large_list (bool, optional): if True, the series_words is a hydrated list of tokens (comma seperated format). Defaults to False.

    Returns:
        gensim.corpora.Dictionary: gensim dictionary
    """
    # takes pandas series as input
    # each item of the series is a words/token list
    from gensim.corpora import Dictionary
    import pandas as pd
    import tqdm

    if large_list is False:
        if isinstance(series_words, pd.Series) is not True:
            print('Not a pandas.Series of tokens lists')
            return(None)

        corpus_dictionary = Dictionary(
            [
                doc_tokens
                for doc_tokens in tqdm.tqdm(
                    series_words,
                    desc='Adding vocabulary to dictionary')])
    else:
        if isinstance(series_words, list) is not True:
            print('Not a large list of tokens')
            return(None)
        # large lists have comma seperated lists of tokens,
        # instead of arrays so they need to be dehydrated
        corpus_dictionary = Dictionary()

        for t in tqdm.tqdm(series_words, desc='Adding vocab to dictionary from large list'):
            corpus_dictionary.add_documents(
                [t.split(',')]
            )

    print(corpus_dictionary.num_docs, ' docs added to dictionary ')

    return(corpus_dictionary)

# %%
# most common 10 tokens


def most_common(words, n=10):
    """
    Returnes the most common words in a document

    Args:
        words (list): list of words in a document
        n (int, optional): Top n common words. Defaults to 10.

    Returns:
        list: list of Top n common terms
    """
    from collections import Counter
    bow = Counter(words)
    ncommon = bow.most_common(n)
    return(ncommon)
# %%
