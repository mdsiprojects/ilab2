# %%
import helpers.potus as potus
import helpers.process as process
import helpers.io as pickle_io
import tqdm
# import pandas as pd

# %%
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--experiment', help='experiment folder name', default='tempexp1', required=False)
arg_parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")


args = arg_parser.parse_args()

expfolder = args.experiment
print('experiment folder name: ', expfolder)


# %%
raw_folder = '/home/azureuser/cloudfiles/code/data/raw/potus_speeches/'
speeches_list = potus.list_speeches(raw_folder)
# %%
# extract tokenised words and tokensied sentences from speech
speeches_df = potus.to_df(speeches_list)
# %%
# save potus speeches df as pickle
df_fn = potus.rootFolder(expfolder) + '/speeches_df.pkl'
pickle_io.to_pickle(speeches_df, df_fn)

# %%
