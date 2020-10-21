# %%
import helpers.process as process
from helpers import io as pickle_io
from gensim.models import ldamodel
import tqdm
from helpers import potus
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
arg_parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

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
root_folder = potus.rootFolder(expfolder)
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
speeches_dictionary.filter_extremes(no_below=10, no_above=0.5)

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
    chunksize=100,
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
log_filename = '/home/azureuser/cloudfiles/code/data/processing/potus/experiment/lda_eval_log.pkl'
try:
    eval_log = pickle_io.from_pickle(log_filename)
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

pickle_io.to_pickle(eval_log, log_filename)


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

