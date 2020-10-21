# %%
def tokenize_sentence(text):
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
    from collections import Counter
    bow = Counter(words)
    ncommon = bow.most_common(n)
    return(ncommon)
# %%
