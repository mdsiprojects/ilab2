# %%
import pandas as pd
from helpers import hansard
import helpers.io as pickle_io
import pprint
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
df_fn = root_folder + '/speeches_df_no_text.pkl'
df_probs_fn = root_folder + '/speeches_df_topics_probs.pkl'
topic_probs_fn = root_folder + '/topic_probs.pkl'

# %%
print(f'load df from {df_fn}')
speeches_df = pd.read_pickle(df_fn)
# %%
print(f'load topics probabilities from {topic_probs_fn}')
probs = pickle_io.from_pickle(topic_probs_fn)
# %%
print(f'loaded df, nrows: {len(speeches_df)}')
print(f'loaded probs, nrows: {len(probs)}')
# %%
print('add probs to df')
speeches_df['probs'] = probs
#pprint(speeches_df.head())
# %%
print('check that topics probs add up to 1 for each doc')
speeches_df.apply(lambda x: sum(x['probs']), axis='columns')
# %%
speeches_df.sort_values(by=['date_time'], ascending=True, inplace=True, ignore_index=True)

# %%
print(f'saving df with topics probs to {df_probs_fn}')
speeches_df.to_pickle(df_probs_fn)
# %%
