# %%
import helpers.process as process
from helpers import io as pickle_io
from helpers import kld
import tqdm
from helpers import hansard
import pandas as pd

# %%
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--experiment', help='experiment folder name', default='eval_06', required=False)

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
speeches_df = hansard.getTopicsProbsDf(exp)
# %%
print('Parliment 43 speeches... ')
parliament43_nonzero_speeches = (speeches_df.token_count > 1) & (speeches_df.parliament == 43)
print(f'Parliment 43 speeches: {sum(parliament43_nonzero_speeches):,}')

# %%
df = speeches_df[parliament43_nonzero_speeches]
print(df.date_time.max() - df.date_time.min())
print(df.date_time.min(), ' - ' , df.date_time.max())
# %%
print('load speeches only...')
df_s = hansard.loadSpeechesOnly(exp, parliament43_nonzero_speeches)
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
df = df.join(df_s.set_index(df.index))

# %% save result to file
filename = f'/speeches_df_43par_{exp}_klds_{Tw}_{Nw}.pkl'
df.to_pickle(hansard.rootFolder(exp) + filename)

# %%
