# %%
import helpers.process as process
from helpers import io as pickle_io
from gensim.models import ldamodel
import tqdm
from helpers import hansard
# from gensim.models import CoherenceModel

# %%
# Set up log to external log file
# import logging
# logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Set up log to terminal
import logging
logging.basicConfig(format=logging.BASIC_FORMAT, level=logging.INFO)

# %%
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--experiment', help='experiment folder name', default='tempexp1', required=False)

arg_parser.add_argument('--num_topics', help='number of topics for modelling', default=100, type=int, required=False)
arg_parser.add_argument('--iterations', help='number of modelling iterations', default=50, type=int, required=False)
arg_parser.add_argument('--passes', help='number of passes for modelling', default=4, type=int, required=False)

args = arg_parser.parse_args()

expfolder = args.experiment
num_topics = args.num_topics
iterations = args.iterations
passes = args.passes

print('experiment folder name: ', expfolder)
print('number of topics: ', num_topics)
print('iterations: ', iterations)
print('passes: ', passes)

# %%
root_folder = hansard.rootFolder(expfolder)
print('root_folder: ', root_folder)

bigrams = pickle_io.from_pickle(root_folder + '/bigrams.pkl')


# # %%
# this logic is move to the begining of the pipeline before the speeches df is saved to file
# min_speech_len = 10
# bigrams = [b for b in bigrams if len(b) >= min_speech_len]

# %%
speeches_dictionary = process.make_dictionary(bigrams, large_list=True)

# %%
# AJ
# Filter out words that occur less than <no_below> documents, or more than <no_above> x 100% of the documents.
speeches_dictionary.filter_extremes(no_below=200, no_above=0.5)

# %%
bow = [
    speeches_dictionary.doc2bow(t.split(','))
    for t in tqdm.tqdm(bigrams, desc='Creating BOW')]


bow_fn = root_folder + '/bow.pkl'
print(f'save bow to {bow_fn}')
pickle_io.to_pickle(bow, bow_fn)
# %% [markdown]
# # Build the topic model using gensim LDA model

print('fit ldamodel')
speeches_lda = ldamodel.LdaModel(
    corpus=bow,  # my list of BOW
    id2word=speeches_dictionary,
    # dictionary for resolving ids from corpus to term in vocablulary
    num_topics=num_topics,
    iterations=iterations,
    passes=passes,
    chunksize=50000,
    update_every=2,
    alpha='auto',
    eta='auto',
    per_word_topics=True,
    random_state=100
)
# %%
lda_model_fn = root_folder + '/lda.model'
print(f'save lda model to file {lda_model_fn}')
speeches_lda.save(lda_model_fn)

# %%
# AJ

import numpy as np

top_topics = speeches_lda.top_topics(bow)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
coherences = [t[1] for t in top_topics]

topic_coherence_mean = sum(coherences) / num_topics
topic_coherence_25th = np.quantile(coherences, 0.25)
topic_coherence_50th = np.quantile(coherences, 0.50)
topic_coherence_75th = np.quantile(coherences, 0.75)

print('Average topic coherence: %.4f.' % topic_coherence_mean)

# Topic diversity: what proportion of the top words across each topic are unique to that topic?
top_terms = []

for x in top_topics:
    for y in x[0]:
        top_terms.append(y[1])

num_terms = len(top_terms)
num_unique = len(set(top_terms))

print('Top topic terms: ' + str(num_terms) + ', of which ' + str(num_unique) +' are unique.')
topic_diversity = num_unique / num_terms
print('Topic diversity score: ' + str(topic_diversity))


from pprint import pprint
pprint(top_topics)


# now save the results to a log pkl

import pickle

try:
    eval_log = pickle_io.from_pickle('/home/azureuser/cloudfiles/code/data/processing/hansard/experiment/lda_eval_log.pkl')
except (OSError, IOError) as e:
    eval_log = []

new_eval_log_entry = {'experiment': expfolder, 
'num_topics': num_topics, 'iterations': iterations, 'passes': passes,
'topic_coherence_mean': topic_coherence_mean, 
'topic_coherence_25th': topic_coherence_25th, 
'topic_coherence_50th': topic_coherence_50th, 
'topic_coherence_75th': topic_coherence_75th, 
'topic_diversity': topic_diversity}

eval_log.append(new_eval_log_entry)

pickle_io.to_pickle(eval_log, '/home/azureuser/cloudfiles/code/data/processing/hansard/experiment/lda_eval_log.pkl')


# %%
# # Compute Perplexity
# print(
#     '\nPerplexity: ',
#     speeches_lda.log_perplexity(bow))
# # a measure of how good the model is. lower the better.


# %%
# Compute Coherence Score

# coherence_model_lda = CoherenceModel(
#     model=speeches_lda,
#     texts=list(speeches_df.words_and_bigrams),
#     dictionary=speeches_dictionary,
#     coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ', coherence_lda)
# # %%
# coherence_model_lda.top_topics_as_word_lists(
#     speeches_lda,
#     dictionary=speeches_dictionary)[4]
# %%
speeches_lda.print_topics()


# %% [markdown]
# for each document, calculate topics and store topic probabiities
#
# this is then used to calculate KLD between documents by using
#  their topic probabiities as inputs
#


# # %%
# probs = list()
# topicids = list()

# for i, r in tqdm.tqdm(speeches_df.iterrows(), desc='calculating topic probabilities'):
#     t, p = zip(*speeches_lda.get_document_topics(bow=r.bow, minimum_probability=0))
#     probs.append([round(e + 1e-17, 8) for e in list(p)])
#     topicids.append(list(t))
#     # adding a small fraction of 1e-17 to work around the inf issue
#     # with entropy when calculating the log of 0
#     if sum(probs[i]) < 0:
#         # testing that probabilties always add up to 1,
#         # important for KLD later
#         print(i)

# speeches_df['topics'] = probs


# # %%
# # calculate novelty
# # Novelty = mean ( Sum of KLD(speech, previous speeches) )
# # prepare an ordered list of speeches and topics
# cols = ['speaker', 'file_name', 'title', 'date', 'topics']
# df = speeches_df[cols].sort_values(by='date', ascending=True)
# from scipy.special import kl_div
# from scipy.stats import entropy
# import numpy as np


# x, y = df.topics[0], df.topics[1]
# kld1 = sum(kl_div(x, y))
# kld2 = entropy(pk=x, qk=y)

# # entropy 0 -> two distributions are the same
# # entrpopy

# Nw = 10  # window of prior speeches
# Tw = 10  # window of future speeches

# novelty = list()
# transience = list()
# resonance = list()
# for i, x in enumerate(df.topics):
#     if i < Nw:
#         # there is not enough previous items
#         Ni = 0
#     else:
#         klds = list()
#         for d in range(Nw):
#             y = df.topics[i - d]
#             k = entropy(pk=x, qk=y)
#             klds.append(k)
#         Ni = np.mean(klds)

#     novelty.append(Ni)

#     if i > len(df.topics) - Tw:
#         Ti = 0
#     else:
#         klds = list()
#         for d in range(Tw):
#             y = df.topics[i + d]
#             k = entropy(pk=x, qk=y)
#             klds.append(k)
#         Ti = np.mean(klds)

#     transience.append(Ti)

#     Ri = Ni - Ti
#     resonance.append(Ri)


# import matplotlib.pylab as plt
# plt.plot(novelty)
# plt.show()
# plt.plot(transience)
# plt.show()
# plt.plot(resonance)
# plt.show()

# # %% [markdown]
# # # TESTS
# # speeches_lda.top_topics()
# # access dictionary
# speeches_lda.id2word.doc2idx(['slave'])
# # %%
# print('num of topics ', speeches_lda.num_topics)
# print('num of terms ', speeches_lda.num_terms)

# # %%
# topics = speeches_lda.get_document_topics(bow=speeches_df.bow[0], minimum_probability=0)
# print(len(topics))
# t, prob = zip(*topics)
# import pandas as pd
# topics_df = pd.DataFrame(topics, columns=['topicid', 'probability'])
# topics_df['topic_terms'] = list(topics_df.topicid.agg({'topicid': speeches_lda.get_topic_terms}))

# # topics_df.topicid.apply(speeches_lda.get_topic_terms)
# # topics_df['topic_terms'] = speeches_lda.get_topic_terms(list(topics_df.topicid))
# # print('DONE!')


# # %%
# #  print topics and top terms in topic
# def print_topic(topicid, n):
#     topic_terms = speeches_lda.get_topic_terms(topicid, topn=n)
#     terms = list()
#     probs = list()
#     for termid, prob in topic_terms:
#         term = speeches_lda.id2word.get(termid)
#         terms.append(term)
#         probs.append(prob)
#     terms = ' '.join(terms)
#     print(f'Topic {topicid:02}: {terms}')


# for i in range(speeches_lda.num_topics):
#     print_topic(i, n=5)

