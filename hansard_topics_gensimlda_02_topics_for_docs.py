# %%
import helpers.process as process
from helpers import io as pickle_io
from gensim.models import ldamodel
from gensim.models import CoherenceModel
#import pyLDAvis
import tqdm
from helpers import hansard
import gc
import logging
import numpy as np
logging.basicConfig(format=logging.BASIC_FORMAT, level=logging.INFO)

# %%
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--experiment', help='experiment folder name', default='tempexp1', required=False)

args = arg_parser.parse_args()

expfolder = args.experiment


print('experiment folder name: ', expfolder)

# %%
root_folder = hansard.rootFolder(expfolder)
print('root_folder: ', root_folder)

# %%
lda_model_fn = root_folder + '/lda.model'
print(f'load lda model from {lda_model_fn}')
speeches_lda = ldamodel.LdaModel.load(lda_model_fn)

bow_fn = root_folder + '/bow.pkl'
print(f'load bow from  {bow_fn}')
bow = pickle_io.from_pickle(bow_fn)

gc.collect()
# %% [markdown]
# ## for each document, calculate topics and store topic probabiities
#
# this is then used to calculate KLD between documents by using
#  their topic probabiities as inputs
#

# %%
probs = list()

# topicids = list(), we dont need list of topic ids, they are allways 0 to 99

for i, b in tqdm.tqdm(enumerate(bow), desc='calculating topic probabilities'):
    t, p = zip(*speeches_lda.get_document_topics(bow=b, minimum_probability=0, per_word_topics=False))
    # probs.append([round(e + 1e-17, 8) for e in list(p)])
    probs.append(
        np.round(
            np.add(p, 1e-17),
            8))

# %%

# save topic probabiilties to file
topic_probs_fn = root_folder + '/topic_probs.pkl'
print(f'print topics probabiilties to {topic_probs_fn}')
pickle_io.to_pickle(probs, topic_probs_fn)
