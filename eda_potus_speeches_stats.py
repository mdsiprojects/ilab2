# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import glob
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
# from datetime import datetime
# import dateparser
# import datefinder
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt

import helpers.potus as read

# %%
root_folder = '/home/azureuser/cloudfiles/code/data/raw/potus_speeches/'
list_speeches = glob.glob(pathname=root_folder + '/**/*.txt', recursive=True)
len(list_speeches)
speech_file = list_speeches[100]
speech_file
speaker_name_re = re.compile(pattern=r"^.*_speeches/([a-z]+).*")
# speaker_name = speaker_name_re.match(speech_file).groups()[0]


speeches_list = [(speaker_name_re.match(f).groups()[0], f, read.parse_file(f)) for f in tqdm(list_speeches)]
len(speeches_list)

# %%
speeches_df = pd.DataFrame(speeches_list, columns=['speaker', 'file_path', 'raw_tuple'])
speeches_df['title'], speeches_df['date'], speeches_df['speech'] = zip(*speeches_df.raw_tuple)

reTokeniser = RegexpTokenizer(r'\w+|$[0-9]+|\S+&[^.<>]')  # , gaps=True)
speeches_df['token_count'] = speeches_df.speech.apply(lambda s: len(reTokeniser.tokenize(s)))


# %%
# plotting
eda_df = speeches_df[['speaker', 'title', 'date', 'token_count']]
eda_df.describe()
speakers = eda_df.speaker.unique()
print(eda_df.speaker.unique())
print(eda_df.speaker.nunique())
eda_df.speaker.value_counts().plot.barh()
# %%
# by speaker:
#   first, last date
#   count of speeches
#   count of tokens
#   distribution of tokens
byspeaker_df = eda_df.groupby('speaker').agg(
    {
        'date': ['min', 'max', ('duration', lambda d: max(d) - min(d))],
        'token_count': [sum, 'mean', 'std', min, max],
        'speaker': [('speeches_count', 'count')]
    }
)
# byspeaker_df.reset_index(inplace=True)
flatten_column_names = ['_'.join(col).strip('_') for col in byspeaker_df.columns]
byspeaker_df.columns = flatten_column_names
byspeaker_df['days'] = byspeaker_df.date_duration.dt.days
byspeaker_df['months'] = round(byspeaker_df.date_duration / np.timedelta64(1, 'M'), 0)
byspeaker_df['years'] = round(byspeaker_df.date_duration / np.timedelta64(1, 'Y'), 0)
byspeaker_df.head()

# %%
byspeaker_df.token_count_sum.plot.box()

# %%

# %%
eda_df.boxplot(column=['token_count'], by=['speaker'])

# %%
# byspeaker_df.speaker_speeches_count.plot.density()
byspeaker_df.speaker_speeches_count.plot.box()

# %%
# byspeaker_df.nlargest(5,'speaker_speeches_count')
byspeaker_df.speaker_speeches_count.nlargest(10)

# %%
byspeaker_df.nlargest(10, 'speaker_speeches_count')[['speaker_speeches_count']]\
    .sort_values('speaker_speeches_count')\
    .plot.barh()

print(
    byspeaker_df.nlargest(
        10,
        'speaker_speeches_count'
    )[['speaker_speeches_count']].sort_values(
        'speaker_speeches_count', ascending=False
    ))

top_10_stats = byspeaker_df.merge(
    right=byspeaker_df.speaker_speeches_count.nlargest(10).reset_index(),
    how='inner',
    on='speaker',
    suffixes=('', '_r')
)

top_10_stats = top_10_stats[['speaker', 'date_min', 'date_max', 'years', 'speaker_speeches_count', 'token_count_sum']]\
    .sort_values('speaker_speeches_count', ascending=False)

top_10_stats['speaches_per_year'] = round(top_10_stats.speaker_speeches_count / top_10_stats.years)

top_10_stats
# %%
# byspeaker_df.nsmallest(5,'speaker_speeches_count')
byspeaker_df.speaker_speeches_count.nsmallest(5)
# %%
# longer than 8 years
# %%
# byspeaker_df.date_duration.plot.hist(bins=100)
byspeaker_df.years.plot.box()
byspeaker_df.years.describe()
print(byspeaker_df[['speaker_speeches_count', 'years']].query('years > 8').sort_values('years', ascending=False))


over_8_years = eda_df.merge(

    right=byspeaker_df[['speaker_speeches_count', 'years']].query('years > 8').reset_index(),
    how='inner',
    on='speaker',
    suffixes=('_l', '_r')
)

over_8_years.date = pd.to_datetime(over_8_years.date)
over_8_years.set_index('date', inplace=True)

over_8_years_speaker_list = over_8_years.speaker.unique()

for speaker in over_8_years_speaker_list:
    speakeryears = int(byspeaker_df.query(f'speaker=="{speaker}"').years[0])
    resampled_df = over_8_years.query(f'speaker=="{speaker}"').resample('QS').sum()
    resampled_df['token_count'].plot()
    plt.title(label=f'token count\n{speaker}\n{speakeryears} years')
    plt.show()
    resampled_df['speaker_speeches_count'].plot(color='green')
    plt.title(label=f'speeches count\n{speaker}\n{speakeryears} years')
    plt.show()


# %%
byspeaker_df.plot.scatter(x='years', y='speaker_speeches_count')
# %%
# explore the top 10 speaks
top_10 = byspeaker_df.speaker_speeches_count.nlargest(10).reset_index()
print('obama' in top_10['speaker'].values)

top_10_line = eda_df.merge(
    right=top_10,
    how='inner',
    on='speaker',
    suffixes=('_l', '_r')
)


# %%
top_10_line.pivot_table(index='date', columns='speaker', values=['token_count'], aggfunc=sum).plot()

# %%
top_10_line.date = pd.to_datetime(top_10_line.date)
top_10_line.set_index('date', inplace=True)
top_10_line_qtr = top_10_line.resample('QS').sum()
# %%
for speaker in top_10.speaker:
    resampled_df = top_10_line.query(f'speaker=="{speaker}"').resample('QS').sum()
    resampled_df['token_count'].plot()
    plt.title(label=f'{speaker}: token_count')
    plt.show()
    resampled_df['speaker_speeches_count'].plot(color='green')
    plt.title(label=f'{speaker}: speaches')
    plt.show()

# %%
