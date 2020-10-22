"""
Text loading and parsing functions for the Australian Parliament Hansard corpus.
"""

from pickle import FALSE
import pandas
from pandas.core import series


def rootFolder(exp=''):
    """
    Get root folder path for Hansard modeling experiments
    Args:
        exp (str, optional): name of experiment. Defaults to ''.

    Returns:
        str: full path for a given experiment
    """
    root_folder = '/home/azureuser/cloudfiles/code/data/processing/hansard/experiment'

    if (len(exp) > 0):
        if (exp.find('/') == 0):
            exp = exp[1:]
        import os.path
        root_folder = os.path.join(root_folder, exp)

        if not os.path.exists(root_folder):
            print(f'Directory {root_folder} does not exist.. ')
            print(f'.. create direcotry')
            os.makedirs(root_folder)

    return(root_folder)


def getTopicsProbsDf(exp='', withCounts=True):
    """
    Load list of topic probabilities for Hansard speeches, the list maintains the same order of speeches as the speeches dataframe in the experiment folder

    Args:
        exp (str, optional): name of experiment. Defaults to ''.
        withCounts (bool, optional): include count of tokens for each speech. Defaults to True.

    Returns:
        pandas.DataFrame: data frame with a column for topic probabiilties, and optional column for token counts
    """
    import pandas as pd
    root_folder = rootFolder(exp)
    df_probs_fn = root_folder + '/speeches_df_topics_probs.pkl'
    print(f'loading df from {df_probs_fn}...')
    df = pd.read_pickle(df_probs_fn)
    print(f'loaded df.')

    if withCounts is True:
        print('loading counts of tokens...')
        speeches_ngrams = getNgramsList(exp)
        # df['token_count'] = list(map(len, speeches_ngrams))
        df['token_count'] = list(map(lambda s: len(s.split(',')), speeches_ngrams))
        print('loaded counts of tokens.')

    return(df)


def loadSpeechesOnly(exp='', slicer=None):
    """
    Load text for speeches.

    Details:
        slicer argument is a series of True and False values, it could be used to filter the list of speeches. One common use for this argument is to filter speeches by party

    Args:
        exp (str, optional): name of experiment. Defaults to ''.
        slicer (pandas.Series, optional): Pandas series for filter speeches. Defaults to None.

    Returns:
        pandas.DataFrame: a dataframe of text for hansard speeches, optionally filtered by input slicer
    """
    import pandas as pd
    root_folder = rootFolder(exp)
    df_speeches_fn = root_folder + '/speeches_df_only_text.pkl'
    print(f'loading df from {df_speeches_fn}...')
    df = pd.read_pickle(df_speeches_fn)
    print(f'loaded df. {df.shape}')

    if slicer is not None:
        print('slicing')
        if isinstance(slicer, pd.Series):
            slicer = slicer.values
        df = df[slicer]
        print(f'df after slicing. {df.shape}')
    return(df)

def getNgramsList(exp=''):
    """
    Load tokens (and bigrams) for the Hansard dataset, the list maintains the same order of speeches as the speeches dataframe in the experiment folder

    Args:
        exp (str, optional): name of experiment. Defaults to ''.

    Returns:
        list: list of tokens for all speeches
    """
    import helpers.io as pickle_io
    root_folder = rootFolder(exp)
    fn = root_folder + '/bigrams.pkl'
    print(f'loading bigrams from {fn}')
    obj = pickle_io.from_pickle(fn)
    print(f'loaded bigrams.')
    return(obj)

