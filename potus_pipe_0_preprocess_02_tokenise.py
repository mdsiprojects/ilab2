# %%
import sys
import pandas as pd
import helpers.potus as potus
import helpers.process as process
from helpers import io
import datetime

# %%
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--experiment', help='experiment folder name', default='tempexp1', required=False)
arg_parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
args = arg_parser.parse_args()

expfolder = args.experiment

print('experiment folder name: ', expfolder)

# %%
root_folder = potus.rootFolder(expfolder)
print('root_folder: ', root_folder)

# %%
print('reading pickle file from: ', root_folder + '/speeches_df.pkl')
speeches_df = pd.read_pickle(root_folder + '/speeches_df.pkl')
print('mem usage: ', speeches_df.memory_usage(deep=True).sum() / 1e+9)
io.print_sys_stats()
# %%
text_only = list(speeches_df.speech)
count_speeches = len(text_only)
print(type(text_only))

# %%
print('delete data frame')
del(speeches_df)
import gc
gc.collect()
# %%
io.print_sys_stats()
time_start = datetime.datetime.now()
print(f'{time_start.strftime("%H:%M:%S")}: Tokenising Speeches...')


# %%
addl_stopwords = []
j = 0
for i, t in enumerate(text_only):
    text_only[i] = ','.join(process.tokenize_word(t, addl_stopwords=addl_stopwords))
    # print progress
    j += 1
    if ((j % 50) == 0):
        currentDT = datetime.datetime.now()
        print(f'{currentDT.strftime("%H:%M:%S")}: processed {j:,} out of {count_speeches:,} speeches')
        io.print_sys_stats()


# %%
time_end = datetime.datetime.now()
print(f'{time_end.strftime("%H:%M:%S")}: Finished tokenising')
print(f'processing time: {(time_end - time_start)}')
io.print_sys_stats()

# %%
print('size of word list: ', sys.getsizeof(text_only)/1e+9)

# %%
gc.collect()
# %%
print('saving word list to  pickle file: ', root_folder + '/word_list.pkl')
io.to_pickle(obj=text_only, file_name=root_folder + '/word_list.pkl')
