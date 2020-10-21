# %%
import pandas as pd
import helpers.hansard as hansard
import helpers.process as process
from helpers import io
import sys
import datetime
import gc
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
print('reading pickle file from: ', root_folder + '/word_list.pkl')
words = pd.read_pickle(root_folder + '/word_list.pkl')
count_speeches = len(words)
print('mem usage: ', sys.getsizeof(words) / 1e+9)


# %%
print('dehydrate word list ')
word_list = list()
for w in words:
    word_list.append(w.split(','))
# %%
print('delete words object')
del(words)
gc.collect()

# %%
io.print_sys_stats()
time_start = datetime.datetime.now()
print(f'{time_start.strftime("%H:%M:%S")}: bigram Speeches...')

# %%
print(f'{datetime.datetime.now().strftime("%H:%M:%S")}: prepare ngram_module')
ngram_module = process.get_ngram_module(word_list=word_list)
print(f'{datetime.datetime.now().strftime("%H:%M:%S")}: created ngram_module')
io.print_sys_stats()

# %%
print('create bigrams')
j = 0
for i, w in enumerate(word_list):
    word_list[i] = ','.join(ngram_module[w])
    # print progress
    j += 1
    if ((j % 1000) == 0):
        currentDT = datetime.datetime.now()
        print(f'{currentDT.strftime("%H:%M:%S")}: processed {j:,} out of {count_speeches:,} speeches')
        io.print_sys_stats()


# %%
time_end = datetime.datetime.now()
print(f'{time_end.strftime("%H:%M:%S")}: Finished creating bigrams')
print(f'processing time: {(time_end - time_start)}')
io.print_sys_stats()


# %%
print('saving bigrams to pickle file: ', root_folder + '/bigrams.pkl')
io.to_pickle(obj=word_list, file_name=root_folder + '/bigrams.pkl')
