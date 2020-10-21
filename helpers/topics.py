# %%
#  print topics and top terms in topic
def get_topic(lda_model, topicid, n, print_prob=False):
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
    terms, probs, s = get_topic(lda_model, topicid, n, print_prob)

    print(s)

    return(terms, probs, s)


# %%
def get_top_topics_from_df(result_model, speech_topic_probs, ntopics=5, nterms=4):
    import numpy as np
    ret = list()
    for topic_index in np.flip(speech_topic_probs.argsort())[:ntopics]:
        terms, probs, s = get_topic(result_model, topic_index, n=nterms, print_prob=False)
        s = f'{100*speech_topic_probs[topic_index]:.1f}% - {s}'
        ret.append(s)
    return(ret)
