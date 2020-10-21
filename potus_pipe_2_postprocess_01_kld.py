# %%
import helpers.process as process
from helpers import io as pickle_io
from helpers import kld
import tqdm
from helpers import potus
import pandas as pd

# %%
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--experiment', help='experiment folder name', default='tempexp1', required=False)

arg_parser.add_argument('--Nw', help='novelty window size', default=50, type=int, required=False)
arg_parser.add_argument('--Tw', help='transience window size', default=50, type=int, required=False)
arg_parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

args = arg_parser.parse_args()

exp = args.experiment
Nw = args.Nw
Tw = args.Tw


print('Post Process: KLD calculations')
print('experiment folder name: ', exp)
print('novelty window size: ', Nw)
print('transience window size: ', Tw)
# %%
speeches_df = potus.getTopicsProbsDf(exp)
# %%
df = speeches_df.copy()
print(df.date.max() - df.date.min())
print(df.date.min(), ' - ' , df.date.max())
# # %%
# print('load speeches only...')
# df_s = nonononono.loadSpeechesOnly(exp, parliament43_nonzero_speeches)
# %%
print('calculating KLD values...')
# df = df.iloc[:1000]
novelty, transience, resonance = kld.calcKLD(df.probs, Nw=Nw, Tw=Tw)
# %%

# df.drop(columns=['novelty', 'transience', 'resonance', 'text' ], inplace=True)
print('adding kld and speech text to df')
df['novelty'] = pd.Series(novelty, index=df.index)
df['transience'] = pd.Series(transience, index=df.index)
df['resonance'] = pd.Series(resonance, index=df.index)


# %%
# df = df.join(df_s.set_index(df.index))

# %% save result to file
filename = f'/speeches_dfr_{exp}_klds_{Tw}_{Nw}.pkl'
df.to_pickle(potus.rootFolder(exp) + filename)

# %%
