# Module `helpers.hansard`

Text loading and parsing functions for the Australian Parliament Hansard
corpus.

## Functions

` def getNgramsList(exp='') `

    

Load tokens (and bigrams) for the Hansard dataset, the list maintains the same
order of speeches as the speeches dataframe in the experiment folder

## Args

**`exp`** : `str`, optional

    name of experiment. Defaults to ''.

## Returns

`list`

    list of tokens for all speeches

` def getTopicsProbsDf(exp='', withCounts=True) `

    

Load list of topic probabilities for Hansard speeches, the list maintains the
same order of speeches as the speeches dataframe in the experiment folder

## Args

**`exp`** : `str`, optional

    name of experiment. Defaults to ''.
**`withCounts`** : `bool`, optional

    include count of tokens for each speech. Defaults to True.

## Returns

`pandas.DataFrame`

    data frame with a column for topic probabiilties, and optional column for token counts

` def loadSpeechesOnly(exp='', slicer=None) `

    

Load text for speeches.

## Details

slicer argument is a series of True and False values, it could be used to
filter the list of speeches. One common use for this argument is to filter
speeches by party

## Args

**`exp`** : `str`, optional

    name of experiment. Defaults to ''.
**`slicer`** : `pandas.Series`, optional

    Pandas series for filter speeches. Defaults to None.

## Returns

`pandas.DataFrame`

    a dataframe of text for hansard speeches, optionally filtered by input slicer

` def rootFolder(exp='') `

    

Get root folder path for Hansard modeling experiments

## Args

**`exp`** : `str`, optional

    name of experiment. Defaults to ''.

## Returns

`str`

    full path for a given experiment

# Index

  * ### Super-module

    * `[helpers](index.html "helpers")`
  * ### Functions

    * `getNgramsList`
    * `getTopicsProbsDf`
    * `loadSpeechesOnly`
    * `rootFolder`

Generated by [pdoc 0.9.1](https://pdoc3.github.io/pdoc).

