"""
Helper functions to query LDA topics model.
"""
# %%
#  print topics and top terms in topic
def get_topic(lda_model, topicid, n, print_prob=False):
    """
    Return terms and probabilities in a topic from an LDA topic model given a topic id

    Args:
        lda_model (gensim.models.ldamodel.LdaModel): trained gensim LDS topic model
        topicid (int): topic ID
        n (int): top n terms in a topic to be returned
        print_prob (bool, optional): generate a formatted string with topic probabilities. Defaults to False.

    Returns:
        tuple(list,list,str): terms, probs, s
    """
    topic_terms = lda_model.get_topic_terms(topicid, topn=n)
    terms = list()
    probs = list()
    for termid, prob in topic_terms:
        term = lda_model.id2word.get(termid)
        terms.append(term)
        probs.append(prob)

    if print_prob is True:
        ts = ' '.join([f'{p[0]}:{p[1]:.3f}' for p in zip(terms, probs)])
    else:
        ts = ' '.join(terms)

    s = f'Topic {topicid:02}: {ts}'

    return(terms, probs, s)


# %%
def print_topic(lda_model, topicid, n, print_prob=False):
    """
    Print top n terms in a topic

    Args:
        lda_model (gensim.models.ldamodel.LdaModel): trained gensim LDS topic model
        topicid (int): topic ID
        n (int): top n terms in a topic to be returned
        print_prob (bool, optional): generate a formatted string with topic probabilities. Defaults to False.
    Returns:
        tuple(list,list,str): terms, probs, s
    """
    terms, probs, s = get_topic(lda_model, topicid, n, print_prob)

    print(s)

    return(terms, probs, s)


# %%
def get_top_topics_from_df(result_model, speech_topic_probs, ntopics=5, nterms=4):
    """
    Get the details of top n topics from a given documents' topic probabilities list
    Args:
        result_model (gensim.models.ldamodel.LdaModel): trained gensim LDS topic model
        speech_topic_probs (list): list of topic probabilities for a document
        ntopics (int, optional): top n topics to returned. Defaults to 5.
        nterms (int, optional): top n terms in a topic. Defaults to 4.
    Returns:
        list: list of strings representing the top n terms and weighting in a given list of topic probabilities
    """
    import numpy as np
    ret = list()
    for topic_index in np.flip(speech_topic_probs.argsort())[:ntopics]:
        terms, probs, s = get_topic(result_model, topic_index, n=nterms, print_prob=False)
        s = f'{100*speech_topic_probs[topic_index]:.1f}% - {s}'
        ret.append(s)
    return(ret)
