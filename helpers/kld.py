
def KLD_N_enumerator(i, probs, Nw=2, Tw=2):
    import pandas as pd
    if isinstance(probs, pd.Series) is True:
        N_enumerator = enumerate(probs.iloc[max(i - Nw,0):i])
    else:
        N_enumerator = enumerate(probs[max(i - Nw,0):i])

    return(N_enumerator)


def KLD_T_enumerator(i, probs, Nw=2, Tw=2):
    import pandas as pd
    if isinstance(probs, pd.Series) is True:
        T_enumerator = enumerate(probs.iloc[i + 1:i + Tw + 1])
    else:
        T_enumerator = enumerate(probs[i + 1:i + Tw + 1])

    return(T_enumerator)


def calcKLD(probs, Nw=100, Tw=100):
    """retruns novelty, transience, resonance
    use as follows: novelty, transience, resonance = hansard.calcKLD(speeches_df[500:510].probs, Nw=3, Tw=9)

    details: Nw and Tw specify maximum window sizes,
        KLD returns zero of no topic probabilities are in the list
        KLD will calculate N or T as long as there is atleast one other document in the window for up to the size of the window

    Args:
        probs (pd.Series or list): must be a list or pd.Series. documents' topic probabilities list, each item in the list is a list of topic probabilities
        Nw (int, optional): prior window size to calculate Novelty. Defaults to 100.
        Tw (int, optional): post window size to calculate transcience. Defaults to 100.
    """
    import tqdm
    import numpy as np
    from scipy.stats import entropy

    novelty = list()
    transience = list()
    resonance = list()

    for i, x in tqdm.tqdm(enumerate(probs), desc='Calculate Novelty, Transience and Resonance'):
        if False:  #i < Nw:
            # there is not enough previous items
            Ni = 0  # need to decide what to set novelt for early items when no history to compare with
        else:
            klds = list()
            n_enumerator = KLD_N_enumerator(i, probs, Nw=Nw)
            for j, p in n_enumerator:
                y = p
                k = entropy(pk=x, qk=y)
                klds.append(k)
            Ni = np.mean(klds) if len(klds) > 0 else 0

        novelty.append(Ni)

        if False:  # i >= len(probs) - Tw:
            Ti = 0
        else:
            klds = list()
            t_enumerator = KLD_T_enumerator(i, probs, Tw=Tw)
            for j, p in t_enumerator:
                y = p
                k = entropy(pk=x, qk=y)
                klds.append(k)
            Ti = np.mean(klds) if len(klds) > 0 else 0

        transience.append(Ti)

        Ri = Ni - Ti
        resonance.append(Ri)

    return(novelty, transience, resonance)
