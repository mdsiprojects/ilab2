# %%
def rootFolder(exp=''):
    root_folder = '/home/azureuser/cloudfiles/code/data/processing/potus/experiment'

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


def parse_file(file_name):
    """Parse a POTUS file and return title, date and speech text

    Args:
        file_name (string): full path to speech text file

    Returns:
        tuple (t,d,s): t: title of speech, d: date of speech, s: full text
    """
    import re
    import datefinder
    with open(file_name, 'r', encoding='utf8') as f:
        file_string = f.read()

        d_iter = re.finditer(r"<date.*\n", file_string)  # [0]
        d_match = next(d_iter)
        d = next(datefinder.find_dates(d_match.group())).date()

        t = re.findall(r"<title.*>", file_string)[0]
        t = t.strip('<>').split('=')[1].strip('"')

        s = file_string[d_match.end():]
    return (t, d, s)


# %%
def list_speeches(root_folder):
    import glob
    import tqdm

    speaker_name_re = glob.re.compile(pattern=r"^.*_speeches/([a-z]+).*")
    speeches_files = glob.glob(
        pathname=root_folder + '/**/*.txt',
        recursive=True)
    speeches_list = [
        (
            speaker_name_re.match(f).groups()[0],
            glob.fnmatch.posixpath.basename(f),
            f,
            parse_file(f)) for f in tqdm.tqdm(
                speeches_files,
                desc='Parsing potus raw speeches')]
    return (speeches_list)


# %%
def to_df(speeches_list):
    import pandas as pd
    df = pd.DataFrame(speeches_list, columns=['speaker', 'file_name', 'file_path', 'raw_tuple'])
    df['title'], df['date'], df['speech'] = zip(*df.raw_tuple)
    # df['speech_len'] = df['speech'].agg({'speech': len})
    df = df.join(df.speech.agg(len), rsuffix='_len')
    return (df)

# %%
def getTopicsProbsDf(exp='', withCounts=True):
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


def getNgramsList(exp=''):
    import helpers.io as pickle_io
    root_folder = rootFolder(exp)
    fn = root_folder + '/bigrams.pkl'
    print(f'loading bigrams from {fn}')
    obj = pickle_io.from_pickle(fn)
    print('loaded bigrams.')
    return(obj)


def gridExpResult(exp, scales=[25, 125], trim=0):
    import pandas as pd
    import glob
    import os
    import re
    from gensim.models import ldamodel
    import helpers.io as pickle_io

    exp_folder = rootFolder(exp)
    exp_kld_files = exp_folder + '/' + 'speeches_dfr_*.pkl'

    # load lda model and BoW
    lda_model_fn = exp_folder + '/lda.model'
    print(f'load lda model from {lda_model_fn}')
    speeches_lda = ldamodel.LdaModel.load(lda_model_fn)

    bow_fn = exp_folder + '/bow.pkl'
    print(f'load bow from  {bow_fn}')
    bow = pickle_io.from_pickle(bow_fn)

    # load kld results df
    files = glob.glob(exp_kld_files)

    kld_settings = list()
    df = pd.DataFrame()
    for fn in files:
        print(fn)
        m = re.findall(r'(\d+)', os.path.basename(fn))
        topics, iterations, passes, Nw, Tw = m

        if int(Nw) in scales:
            print(f't: {topics}, Nw: {Nw}, Tw:{Tw}')
            kld_settings.append({
                'kld_filename': os.path.basename(fn),
                'topics': int(topics),
                'Nw': int(Nw),
                'Tw': int(Tw),
                'path': fn
            })

            dft = pd.read_pickle(fn)
            dft['kld_filename'] = os.path.basename(fn)
            print(len(dft), ' before trim')

            dft['speech_id'] = dft.apply(lambda r: f'{r.speaker}_{r.name:03d}', axis=1)

            if trim > 0:
                df = df.append(dft[trim:-trim])
            else:
                df = df.append(dft)

    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    df.sort_values(by='date', ascending=True, inplace=True)
    df = pd.merge(df, pd.DataFrame(kld_settings), how='left', on='kld_filename', indicator=True)

    # combine all results in return object
    ret = objectview({
        'df': df,
        'settings': pd.DataFrame(kld_settings),
        'model': speeches_lda,
        'bow': bow})

    return(ret)
