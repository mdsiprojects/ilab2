# %%
from glob import glob
import pandas as pd
import numpy as np
import helpers.io as pickle_io
import helpers.hansard as hansard


# %%
import sys
print(sys.version)

# %%
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--experiment', help='experiment folder name', default='tempexp1', required=False)
arg_parser.add_argument('--filename', help='speeches list pkl file path', default='/home/azureuser/cloudfiles/code/Users/Shared/hansard/corpus/hansard_speech_records_parse_20_10_03-01_19_24.pkl', required=False)
arg_parser.add_argument('--n_topics', help='number of topics', default=100, type=int, required=False)
arg_parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")


args = arg_parser.parse_args()

expfolder = args.experiment
speeches_pkl_filename = args.filename
print('experiment folder name: ', expfolder)
print('file name: ', speeches_pkl_filename)


# %%
import os
os.getcwd()
pd.__version__

# %%
root_folder = hansard.rootFolder(expfolder)
print('root_folder: ', root_folder)

# list_pickles = glob(root_folder + '/hansard_speech_records_test_20_09_22-18_22_10.pkl')
list_pickles = glob(speeches_pkl_filename)
fn = list_pickles[0]
column_names = [
    'date_time',
    'chamber',
    'parliament',
    'session',
    'period',
    'page',
    'name',
    'name_id',
    'party',
    'in_gov',
    'electorate',
    'first_speech',
    'context',
    'context_title',
    'context_type',
    'speech_type', # AJ
    'text'
]
column_names_no_text = column_names[:-1]

# %%
speeches_list = pickle_io.from_pickle(fn)

# %%
speeches_df = pd.DataFrame(
    speeches_list,
    columns=column_names)

# exclude speeches by the chair - AJ
speeches_df = speeches_df[~speeches_df['name'].str.contains('SPEAKER') & ~speeches_df['name'].str.contains('PRESIDENT')]

size_before = speeches_df.memory_usage(deep=True).sum() / 1e+9

speeches_df.chamber = pd.Categorical(speeches_df.chamber)
speeches_df.parliament = pd.Categorical(speeches_df.parliament.astype(int), ordered=True)
speeches_df.date_time = pd.to_datetime(speeches_df.date_time, unit='D')
speeches_df.session = pd.Categorical(speeches_df.session.astype(int), ordered=True)
speeches_df.period = pd.Categorical(speeches_df.period.astype(int), ordered=True)
speeches_df.page = speeches_df.page.astype(np.uint8)
speeches_df.name = pd.Categorical(speeches_df.name)
speeches_df.name_id = pd.Categorical(speeches_df.name_id)

speeches_df.party = pd.Categorical(speeches_df.party)
speeches_df.in_gov = speeches_df.in_gov.astype(bool)
speeches_df.electorate = pd.Categorical(speeches_df.electorate)

speeches_df.first_speech = pd.Categorical(speeches_df.first_speech)
speeches_df.context = pd.Categorical(speeches_df.context)
speeches_df.context_title = pd.Categorical(speeches_df.context_title)
speeches_df.context_type = pd.Categorical(speeches_df.context_type)

# exclude speeches based on context_type; these are deemed to not represent 'debate'
context_exclusions = ['bills', 'questions without notice', 'answers to questions on notice', 'adjournment', 'questions in writing', 'committees', 'statements by members', 'answers to questions in writing', 'motions', 'matters of public importance', "private members' business", 'miscellaneous', 'business', 'questions without notice: take note of answers', 'questions on notice', 'ministerial statements', 'petitions', 'matters of public interest', 'statements on indulgence', 'matters of urgency', 'questions without notice: additional answers', 'statements by senators', 'statements', 'petitions', 'first speech', 'private membersâ business', 'resolutions of the senate', 'answers to questions on notice', 'special adjournment']
speeches_df = speeches_df[speeches_df['context_type'].isin(context_exclusions)]

size_after = speeches_df.memory_usage(deep=True).sum() / 1e+9
print(f'size before: {size_before} GB, size after: {size_after} GB')

# %%
min_speech_length = 250
# df = speeches_df[speeches_df.text.str.len() <= min_speech_length]
speeches_df = speeches_df[speeches_df.text.str.len() >= min_speech_length]
# %%
speeches_df.to_pickle(root_folder + '/speeches_df.pkl')

# %%
speeches_df[column_names_no_text].to_pickle(root_folder + '/speeches_df_no_text.pkl')
# %%
speeches_df[['text']].to_pickle(root_folder + '/speeches_df_only_text.pkl')
# %%
