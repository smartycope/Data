import pandas as pd
from warnings import warn
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import seaborn as sns
from scipy.stats import entropy as _entropy
import matplotlib.pyplot as plt
from typing import Optional, Any, Tuple, List, Iterable, Dict, Union
import numpy as np
from enum import Enum
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay
import ipywidgets as widgets
from typing import Union, Callable, Iterable
from collections import OrderedDict
from sympy import Integer, Float
from IPython.display import clear_output, display
from math import log, e

# If there's mutliple modes, how do we want to choose one? Used in _cleanColumn
# options: 'random', 'first', 'last'
MODE_SELECTION = 'random'

# How small is a "small" dataset
SMALL_DATASET = 1000
HIGH_CARDINALITY = 50
# At what percentage does it become worrisome if that many samples are missing the feature?
ALERT_MISSING = .55
# If the difference between the extreme and the mid extreme > median * this, then it indicates outliers
OUTLIER_THRESHOLD = .5

# np.object_ is a string type (apparently)
_catagoricalTypes = (str, Enum, np.object_, pd.CategoricalDtype, pd.Interval, pd.IntervalDtype, type(np.dtype('O')))

try:
    from Cope import todo
except ImportError:
    todo = lambda *a: print('TODO: ', *a)

def insertSample(df, sample, index=-1):
    """ Because theres not a function for this? """
    df.loc[index - .5] = sample
    return df.sort_index().reset_index(drop=True)

def ensureIterable(obj, useList=False):
    if not isiterable(obj):
        return [obj, ] if useList else (obj, )
    else:
        return obj

def normalizePercentage(p, error='Percentage is of the wrong type (int or float expected)'):
    if isinstance(p, (int, Integer)):
        return p / 100
    elif isinstance(p, (float, Float)):
        return p
    elif isinstance(p, bool):
        if p == True:
            return 1.
        else:
            return 0.
    else:
        if error is not None:
            raise TypeError(error)

def isiterable(obj, includeStr=False):
    return isinstance(obj, Iterable) and (type(obj) is not str if not includeStr else True)

def sort_dict_by_value_length(d):
    return dict(sorted(d.items(), key=lambda item: len(item[1])))

def isQuantatative(s: pd.Series):
    return not isCatagorical(s)

def isCatagorical(s: pd.Series):
    # This is a total hack, but I'm out of ideas                        \/
    return len(s) and (isinstance(s[0], _catagoricalTypes) or str(s.dtype) == 'object')

def splitDataFrameTypes(df, outOf=None):
    cat = []
    quant = []

    if outOf is None:
        outOf = list(df.columns)
    else:
        ensureIterable(outOf)

    for col in outOf:
        if isCatagorical(df[col]):
            cat.append(col)
        else:
            quant.append(col)

    return cat, quant

def catagorical(df, outOf=None):
    return splitDataFrameTypes(df, outOf)[0]

def quantatative(df, outOf=None):
    return splitDataFrameTypes(df, outOf)[1]

def missingSummary(df, thresh=.6):
    table = df.isnull().sum()/len(df)
    return table[table >= thresh]

def significantCorrelations(df, thresh=.5):
    names = df.columns
    cor = df.corr()
    # Find the significant correlations
    pos = cor[cor >=  thresh]
    neg = cor[cor <= -thresh]
    # Convert the NaN's to 0's (because math)
    pos[pos.isna()] = 0
    neg[neg.isna()] = 0
    # We can add these, because there will never be both a positive and negative corellation at the same time
    arr = pos + neg
    # Remove the obvious correlations along the diagonal
    l, w = cor.shape
    assert l == w, 'Somehow the correlation matrix isnt square?'
    # np.fill_diagonal(arr, 0)
    arr[np.eye(w) == 1] = 0
    # Remove the rows and columns which don't have any significant correlations (all 0's)
    arr = arr.loc[:, (arr != 0).any(axis=0)]
    arr = arr[(arr != 0).any(axis=1)]
    # Because the correlations repeat, remove the upper triangular matrix
    arr = np.triu(arr.to_numpy())
    # Get the indecies of the non-zero entries
    nonzero_indices = list(zip(*np.where(arr != 0)))
    rtn = []
    for r, c in nonzero_indices:
        rtn.append((names[r], names[c], arr[r, c]))
    return rtn

def getNiceTypesTable(df, types=None):
    niceTypeNames = {
        np.object_: 'C',
        np.dtype('O'): 'C',
        np.int64: 'Q',
        np.dtype('int64'): 'Q',
        np.dtype('float64'): 'Q',
        pd.CategoricalDtype: 'C',
        pd.Interval: 'C',
        pd.IntervalDtype: 'C',
    }
    # return pd.DataFrame(dict(zip(df.columns, [(['C'] if isinstance(df[i].dtype, _catagoricalTypes) else ['Q']) for i in df.columns])))
    if types is None:
        return pd.DataFrame(dict(zip(df.columns, [([df[i].dtype, 'C'] if isinstance(df[i].dtype, _catagoricalTypes) else [df[i].dtype, 'Q']) for i in df.columns])))
    elif types:
        return pd.DataFrame(dict(zip(df.columns, [[df[i].dtype] for i in df.columns])))
    else:
        return pd.DataFrame(dict(zip(df.columns, [(['C'] if isinstance(df[i].dtype, _catagoricalTypes) else ['Q']) for i in df.columns])))

def _normalize(df, method='default'):
    for col in quantatative(df):
        if method == 'default':
            df[col] = (df[col]-df[col].mean())/df[col].std()
        elif method == 'min-max':
            # display(df[col])
            df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())
        else:
            raise TypeError('Invalid method parameter')

def percentCountPlot(data, feature, target=None, ax=None, title='Percentage of values used in {}'):
    # plt.figure(figsize=(20,10))
    # plt.title(f'Percentage of values used in {feature}')
    Y = data[feature]
    total = float(len(Y))
    ax=sns.countplot(x=feature, data=data, hue=target, ax=ax)
    ax.set_title(title.format(feature))
    for p in ax.patches:
        ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))

    #put 11 ticks (therefore 10 steps), from 0 to the total number of rows in the dataframe
    ax.yaxis.set_ticks(np.linspace(0, total, 11))
    #adjust the ticklabel to the desired format, without changing the position of the ticks.
    ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    # ax.legend(labels=["no","yes"])
    # plt.show()
    return ax

def column_entropy(column:pd.Series, base=e):
    """ This works, but it's slow for some reason? """
    vc = pd.Series(column).value_counts(normalize=True, sort=False)
    return -(vc * np.log(vc)/np.log(base)).sum()

def pretty_2_column_array(a, limit=30):
    card = len(a)
    if card > limit:
        a = a[:limit-1]
        # a.append(f'... ({card - limit - 1} more)')

    offset = max(list(a.index), key=len)
    rtn = ''
    for i in range(len(a)):
        rtn += f'\t{a.index[i]:>{len(offset)}}: {a[i]:.1%}\n'
    return rtn

def pretty_counts(s:pd.Series):
    # rtn = ''
    # for i in s.value_counts(normalize=True, sort=True):
    #     rtn += str(i)
    # rtn = str()
    rtn = pretty_2_column_array(s.value_counts(normalize=True, sort=True))
    return rtn


# The main functions
def quickSummary(data,
                 relevant=None,
                 target=None,
                 notRelevant=None,
                 stats=None,
                 additionalStats=[],
                 missing=True,
                 corr=.5,
                 entropy=None,
                 start='Head'
    ):
    # Parse params and make sure all the params are valid
    assert relevant is None or notRelevant is None, 'Please dont specify both relevant and not relevant columns at the same time'
    assert not isinstance(target, (list, tuple)), 'There can only be 1 target feature'
    assert target is None or target in data.columns, f'Target {target} is not one of the features'

    if relevant is None:
        relevant = list(data.columns)
    if notRelevant is not None:
        for i in ensureIterable(notRelevant):
            assert i in data.columns, f'{i} is not one of the features'
            relevant.remove(i)
    if stats is None:
        stats = ['mean', 'median', 'std', 'min', 'max']
    if stats:
        stats += ensureIterable(additionalStats, True)
    relevant = list(relevant)
    for i in relevant:
        assert i in data.columns, f'{i} is not one of the features'


    # Define variables
    _relevant = data[relevant]
    quant = data[quantatative(data, relevant)]
    cat = data[catagorical(data, relevant)]
    # Just to make sure we're catching all the columns
    # Convert to set so order doesn't matter
    assert set(quantatative(data, relevant) + catagorical(data, relevant)) == set(relevant)
    # Okay, so we have corr here, and we *should* have it in the output() function,
    # but for SOME REASON it and missing aren't in that scope. They have all the
    # OTHER parameters, just not corr and missing. Somehow. SOMEHOW????
    whatTheHeck = (corr, missing)
    max_name_len = len(max(data.columns, key=len))


    # Define widget[s]
    combobox = widgets.Dropdown(
            options=[
                'Description',
                'Features',
                'Head',
                'Stats',
                'Missing',
                'Duplicates',
                'Entropy',
                'Counts',
                'Correlations',
                'General Plots',
                'Custom Plots',
                'Matrix',
                'Alerts',
            ],
            value=start,
            description='Select Summary',
            style={'description_width': 'initial'},

            # title='hello there'
        )
    # This doesn't work
    # combobox.box_style = 'primary'

    # To make this work, this is always there, we just set it to hidden when not
    # under the features page
    featureBox = widgets.Dropdown(
                    options=list(data.columns),
                    value=target if target is not None else list(data.columns)[0],
                    description='Feature',
                )
    featureBox.layout.visibility = 'hidden'
    featureABox = widgets.Dropdown(
        options=list(data.columns),
                    value=target if target is not None else list(data.columns)[0],
                    description='x',
        )
    featureABox.layout.visibility = 'hidden'
    featureBBox = widgets.Dropdown(
        options=list(data.columns),
                    value=target if target is not None else list(data.columns)[0],
                    description='y',
        )
    featureBBox.layout.visibility = 'hidden'

    # All the actual logic
    def output(page, feature, a, b):
        # See baffled comment above
        corr, missing = whatTheHeck
        featureBox.layout.visibility = 'hidden'
        featureABox.layout.visibility = 'hidden'
        featureBBox.layout.visibility = 'hidden'
        # Clear the output (because colab doesn't automatically or something?)
        clear_output(wait=True)

        # match page:
        if page == 'Description':
                print(f'There are {len(data):,} samples, with {len(data.columns)} columns:')
                display(getNiceTypesTable(data))

                print()

                print('The possible values for the Catagorical values:')
                # This is just an overly complicated way to print them all nicely
                for key, value in sort_dict_by_value_length(dict([(c, data[c].unique()) for c in cat])).items():
                    # If it has too high of a cardinality, just print the first few
                    card = len(value)
                    shortened = False
                    if card > 30:
                        shortened = True
                        value = value[:29]

                    print(key + ":")
                    joined_list = ", ".join(value)
                    if len(joined_list) <= 80: # adjust this number as needed
                        print('   ' + joined_list)
                    else:
                        for item in value:
                            print('   ' + item)
                    print(f'... ({card - 29} more catagories)')
        elif page == 'Stats':
                if len(quant):
                    # print('Summary of Quantatative Values:')
                    display(_relevant.agg(dict(zip(quant, [stats]*len(relevant)))))
        elif page == 'Entropy':
                todo('Calculate entropy relative to the target feature')
                # if target is not None:
                # base = e if entropy is not None else entropy
                for c in data.columns:
                    print(f'The entropy of {c:>{max_name_len}} is: {round(_entropy(data[c].value_counts(normalize=True), base=entropy), 3)}')
                    # print(f'The entropy of {c} is: {entropy(data[c], data[target])}')
                # else:
                    # print('Target feature must be provided in order to calculate the entropy')
        elif page == 'Duplicates':
                todo()
        elif page == 'Head':
                display(data.head())
        elif page == 'Counts':
                # This is sorted just so the features with less unique options go first
                for i in sorted(cat, key=lambda c: len(data[c].unique())):
                    print(f'{i} value counts:')

                    if len(data[i].unique()) == len(data[i]):
                        print('\tEvery sample has a unique catagory')
                    else:
                        print(pretty_counts(data[i]))
        elif page == 'Correlations':
                if len(quant):
                    print('Correlations Between Quantatative Values:')
                    if type(corr) is bool:
                        display(quant.corr())
                    elif isinstance(corr, (int, float)):
                        corr = normalizePercentage(corr)
                        # Ignore if they're looking for a negative correlation, just get both
                        corr = abs(corr)
                        _corr = significantCorrelations(quant, corr)
                        if len(_corr):
                            a_len = max([len(i[0]) for i in _corr])
                            b_len = max([len(i[1]) for i in _corr])
                            for a,b,c in _corr:
                                print(f'\t{a:<{a_len}} <-> {b:<{b_len}}: {round(c, 2):+}')
                        else:
                            print(f'\tThere are no correlations greater than {corr:.0%}')
        elif page == 'Missing':
                if len(_relevant):
                    print('Missing Percentages:')
                    if type(missing) is bool:
                        percent = _relevant.isnull().sum()/len(_relevant)*100
                        # This works, but instead I decided to overcomplicate it just so I can indent it
                        # print(percent)
                        print(pretty_2_column_array(percent))
                    elif isinstance(missing, (int, float)):
                        missing = normalizePercentage(missing)
                        _missing = missingSummary(_relevant, missing/100)
                        if len(_missing):
                            display(_missing)
                        else:
                            print(f'\tAll values are missing less than {missing:.0%} of their entries')
                    else:
                        raise TypeError('Missing is a bad type')
        elif page == 'Features':
                todo('Add nice plots here: scatterplots, histograms, and relating to the target feature')
                # TODO: mode[s], std, quantative entropy, catagorical correlations, data.groupby(feature)[target].value_counts(),
                featureBox.layout.visibility = 'visible'

                # Quantative and Catagorical attributes
                group = 'catagorical' if isCatagorical(data[feature]) else 'quantative'
                missing = data[feature].isnull().sum()/len(data[feature])
                shared = f'"{feature}" is {"the target" if feature == target else "a"} {group} feature of type {data[feature].dtype}.\n' \
                            f'{missing:.1%} of it is missing.'

                # Catagorical description
                if isCatagorical(data[feature]):
                    print(shared)
                    print(f'It has an entropy of {_entropy(data[feature].value_counts(normalize=True), base=entropy):.3f}', end=', ')
                    print(f'and a cardinaltiy of {len(data[feature].unique())}')
                    print('Value counts:')
                    print(pretty_counts(data[feature]))

                # Quantative description
                else:
                    correlations = []
                    for a, b, c in significantCorrelations(quant, corr):
                        other = None
                        if a == feature:
                            other = b
                        elif b == feature:
                            other = a
                        if other is not None:
                            correlations.append(f'{other}({c:.1%})')

                    if len(correlations):
                        correlations = 'It correlates with ' + ', '.join(correlations)
                    else:
                        correlations = f'It has no significant (>{corr:.1%}) correlations with any features'

                    print(shared)
                    print(f'It has an average value of {data[feature].mean():,.2f}, and a median of {data[feature].median():,.2f}.')
                    print(f'It has a minimum value of {data[feature].min():,.2f}, and a maximum value of {data[feature].max():,.2f}.')
                    print(correlations)
        elif page == 'General Plots':
                if len(quant):
                    print('Plot of Quantatative Values:')
                    sns.catplot(data=quant)
                    plt.show()
                if len(cat):
                    print('Plot of Catagorical Value Counts:')
                    todo('catagorical (count?) plots')
                    # plt.show()
        elif page == 'Custom Plots':
                featureABox.layout.visibility = 'visible'
                featureBBox.layout.visibility = 'visible'

                graph = sns.scatterplot(x=data[a], y=data[b])
                if isQuantatative(data[a]) and isQuantatative(data[b]):
                    try:
                        graph.set(title=f'Correlation: {data.corr()[a][b]}')
                    except KeyError:
                        print('Cant calculate the correlations of dates for some reason')
                else:
                    # counts = data.groupby(a)[b].value_counts()
                    # print(counts.index.max())
                    # print(counts)
                    graph.set(title=f'Most common together: Todo')

                plt.show()
        elif page == 'Matrix':
                if len(quant):
                    print('Something Something Matrix:')
                    if target in quantatative(data, relevant):
                        sns.pairplot(data=quant, hue=target)
                    else:
                        sns.pairplot(data=quant)
                    plt.show()
        elif page == 'Alerts':
                # TODO:
                # Check that entropy isn't too low
                # check that relative entropy isn't too low
                # check for spikes and plummets
                # high correlations between features

                # Check if our dataset is small
                if data[feature].count() < SMALL_DATASET:
                    print(f"Your dataset isn't very large ({data[feature].count()}<{SMALL_DATASET})")

                # Check the cardinality
                for c in cat:
                    card = len(data[c].unique())
                    if card == 1:
                        print(f'All values in feature "{c}" are the same')
                    elif card >= data[feature].count():
                        print(f'Every value in feature "{c}" is unique, are you sure its not quantatative?')
                    elif card > HIGH_CARDINALITY:
                        print(f'Feature "{c}" has a very high cardinality ({card}>{HIGH_CARDINALITY})')

                # Check we're not missing too many
                for i in data.columns:
                    miss = data[i].isnull().sum()/len(data[i])
                    if miss >= ALERT_MISSING:
                        print(f'Feature {i} is missing a significant portion ({miss}>={ALERT_MISSING})')

                # Check for outliers
                for q in quant:
                    upper = data[q].max() - data[q].quantile(.75)
                    upperMid = data[q].quantile(.75) - data[q].median()
                    if upper - upperMid > OUTLIER_THRESHOLD * data[q].median():
                        print(f'Feature {q:>{max_name_len}} may have some upper outliers', end='   | ')
                        print(f'upper: {upper:>6.1f} | upperMid: {upperMid:>6.1f} | median: {data[q].median():>6.1f} | diff: {upper-upperMid:>6.1f}')

                    lower = data[q].quantile(.25) - data[q].min()
                    lowerMid = data[q].median() - data[q].quantile(.25)
                    if lower -  lowerMid > OUTLIER_THRESHOLD * data[q].median():
                        print(f'Feature {q:>{max_name_len}} may have some lower outliers', end='   | ')
                        print(f'lower: {lower:>6.1f} | lowerMid: {lowerMid:>6.1f} | median: {data[q].median():>6.1f} | diff: {lower-lowerMid:>6.1f}')
        else:
                print('Invalid start option')

    # widgets.interact(output, page=combobox, feature=featureBox)
    ui = widgets.GridBox([combobox, featureABox, featureBox, featureBBox], layout=widgets.Layout(
                grid_template_columns='auto auto',
                grid_row_gap='10px',
                grid_column_gap='100px',
            )
       )
    out = widgets.interactive_output(output, {'page': combobox, 'feature': featureBox, 'a': featureABox, 'b': featureBBox})
    display(ui, out)

def suggestedCleaning(df, target):
    todo('suggestedCleaning')

def _cleanColumn(df, args, column, verbose, ignoreWarnings=False):
    global MODE_SELECTION
    log = lambda s: print('\t' + s) if verbose else None
    missing = np.nan
    if column in df.columns:
        for op, options in args.items():
            if op == 'drop_duplicates':
                if options:
                    warn('drop_duplicates hasnt been implemented yet for induvidual columns. What are you trying to do?')
                    # log(f'Dropping duplicates in {column}')
                    # df[column].drop_duplicates(inplace=True)
            if op == 'replace':
                if options == True:
                    if not ignoreWarnings:
                        raise TypeError(f"Please specify a dict for the replace option")
                if isinstance(options, dict):
                    log(f'Replacing {column}')
                    df[column] = df[column].replace(options)
            if op == 'apply':
                if options == True:
                    if not ignoreWarnings:
                        raise TypeError(f"Please specify a function to apply")
                elif callable(options):
                    log(f'Applying function to {column}')
                    df[column] = df[column].apply(options)
            if op == 'missing_value':
                missing = options
            if op == 'handle_missing':
                without = df.loc[df[column] != missing]
                # match options:
                if options == True:
                        if not ignoreWarnings:
                            raise TypeError(f"Please specify a value or method for the handle_missing option")
                elif options == False:
                        pass
                elif options == 'remove':
                        log(f'Removing all samples with a "{column}" values of "{missing}"')
                        df = without
                elif options == 'mean':
                        if isCatagorical(df[column]):
                            if not ignoreWarnings:
                                raise TypeError(f"Cannot get mean of a catagorical feature")
                            else:
                                continue
                        mean = without[column].mean()
                        log(f'Setting all samples with a "{column}" value of "{missing}" to the mean ({mean:.2})')
                        df.loc[df[column] == missing, column] = mean
                elif options == 'median':
                        if isCatagorical(df[column]):
                            if not ignoreWarnings:
                                raise TypeError(f"Cannot get median of a catagorical feature")
                            else:
                                continue
                        median = without[column].median()
                        log(f'Setting all samples with a "{column}" value of "{missing}" to the median ({median})')
                        df.loc[df[column] == missing, column] = median
                elif options == 'mode':
                        # I'm not sure how else to pick a mode, so just pick one at random
                        if MODE_SELECTION == 'random':
                            mode = random.choice(without[column].mode())
                        elif MODE_SELECTION == 'first':
                            mode = without[column].mode()[0]
                        elif MODE_SELECTION == 'last':
                            mode = without[column].mode()[-1]
                        log(f'Setting all samples with a "{column}" value of "{missing}" to a mode ({mode})')
                        df.loc[df[column] == missing, column] = mode
                elif options == 'random':
                        if isCatagorical(df[column]):
                            log(f'Setting all samples with a "{column}" value of "{missing}" to random catagories')
                            def fill(sample):
                                if sample == missing:
                                    return random.choice(without[column].unique())
                                else:
                                    return sample
                        else:
                            log(f'Setting all samples with a "{column}" value of "{missing}" to random values along a uniform distrobution')
                            def fill(sample):
                                if sample == missing:
                                    return type(sample)(random.uniform(without[column].min(), without[column].max()))
                                else:
                                    return sample

                        df[column] = df[column].apply(fill)
                elif options == 'balanced_random':
                        if isCatagorical(df[column]):
                            log(f'Setting all samples with a "{column}" value of "{missing}" to evenly distributed random catagories')
                            def fill(sample):
                                if sample == missing:
                                    return random.choice(without[column])
                                else:
                                    return sample
                        else:
                            log(f'Setting all samples with a "{column}" value of "{missing}" to random values along a normal distrobution')
                            def fill(sample):
                                if sample == missing:
                                    return type(sample)(random.gauss(without[column].mean(), without[column].std()))
                                else:
                                    return sample
                        df[column] = df[column].apply(fill)
                else:
                        log(f'Setting all samples with a "{column}" value of "{missing}" to {options}')
                        df.loc[df[column] == missing, column] = options
            if op == 'queries':
                if options == True:
                    if not ignoreWarnings:
                        raise TypeError(f"Please specify a queries and values for the queries option")
                elif options == False:
                    continue
                else:
                    try:
                        # If there's just one query, just accept it
                        if len(options) == 2 and type(options[0]) is str:
                            options = [options]

                        for query, replacement in options:
                            # match replacement:
                            if replacement == 'remove':
                                    log(f'Removing all samples where "{query}" is true')
                                    df = df.drop(df.query(query).index)
                            elif replacement == 'mean':
                                    if isCatagorical(df[column]):
                                        if not ignoreWarnings:
                                            raise TypeError(f"Cannot get mean of a catagorical feature")
                                        else:
                                            continue
                                    mean = df[column].mean()
                                    log(f'Setting all samples where {query} is true to the mean of "{column}" ({mean:.2})')
                                    df.loc[df.query(query).index, column] = mean
                            elif replacement == 'median':
                                    if isCatagorical(df[column]):
                                        if not ignoreWarnings:
                                            raise TypeError(f"Cannot get median of a catagorical feature")
                                        else:
                                            continue
                                    median = df[column].median()
                                    log(f'Setting all samples where "{query}" is true to the median of "{column}" ({median})')
                                    df.loc[df.query(query).index, column] = median
                            elif replacement == 'mode':
                                    # I'm not sure how else to pick a mode, so just pick one at random
                                    if MODE_SELECTION == 'random':
                                        mode = random.choice(df[column].mode())
                                    elif MODE_SELECTION == 'first':
                                        mode = df[column].mode()[0]
                                    elif MODE_SELECTION == 'last':
                                        mode = df[column].mode()[-1]
                                    log(f'Setting all samples where "{query}" is true to a mode of "{column}" ({mode})')
                                    df.loc[df.query(query).index, column] = mode
                            elif replacement == 'random':
                                    if isCatagorical(df[column]):
                                        log(f'Setting all samples where "{query}" is true to have random catagories')
                                        def fill(sample):
                                            return random.choice(df[column].unique())
                                    else:
                                        log(f'Setting all samples where "{query}" is true to have random values along a uniform distrobution')
                                        def fill(sample):
                                            return type(sample)(random.uniform(df[column].min(), df[column].max()))

                                    q = df.query(query)
                                    df.loc[q.index, column] = q[column].apply(fill)
                            elif replacement == 'balanced_random':
                                    if isCatagorical(df[column]):
                                        log(f'Setting all samples where "{query}" is true to have evenly distributed random catagories')
                                        def fill(sample):
                                            return random.choice(df[column])
                                    else:
                                        log(f'Setting all samples where "{query}" is true to have random values along a normal distrobution')
                                        def fill(sample):
                                            return type(sample)(random.gauss(df[column].mean(), df[column].std()))

                                    q = df.query(query)
                                    df.loc[q.index, column] = q[column].apply(fill)
                            else:
                                    log(f'Setting all samples where "{query}" is true to have a "{column}" value of  {options}')
                                    df.loc[df.query(query).index, column] = replacement
                    except ValueError:
                        raise TypeError(f"Invalid queries option. It's supposed to be a list of 2 item tuples.")
            if op == 'remove':
                log(f'Removing all samples with a "{column}" value of {options}')
                df = df.loc[df[column] != options]
            if op == 'bin':
                if isCatagorical(df[column]):
                    if not ignoreWarnings:
                        warn(f'The bin option was set on "{column}", which is not quantatative, skipping.')
                        continue
                else:
                    # match options:
                    if options == True:
                            if not ignoreWarnings:
                                raise TypeError(f"Please specify a method for the bin option")
                    elif options[0] == 'frequency':
                            log(f'Binning "{column}" by frequency into {options[1]} bins')
                            df[column] = pd.qcut(df[column], options[1], duplicates='drop')
                    elif options[0] == 'width':
                            log(f'Binning "{column}" by width into {options[1]} bins')
                            raise NotImplementedError('Width binning')
                    elif isinstance(options, (tuple, list)):
                            log(f'Custom binning "{column}" into {len(options)} bins')
                            df[column] = pd.cut(df[column], options)
                    else:
                        raise TypeError(f"Bin option given bad arguement")
            if op == 'normalize':
                if isCatagorical(df[column]):
                    if not ignoreWarnings:
                        warn(f'The normalize option was set on {column}, which is not quantatative, skipping.')
                        continue
                else:
                    # match options:
                    if options == True or options == 'min-max':
                        log(f'Normalizing "{column}" by min-max method')
                        df[column] = (df[column]-df[column].min())/(df[column].max()-df[column].min())
                    elif options == 'range':
                        log(f'Normalizing "{column}" by range method')
                        raise NotImplementedError(f'range normalization doesn\'t work yet')
                        df[column] = (df[column]-df[column].mean())/df[column].std()
                    else:
                        raise TypeError('Invalid normalize argument given')
            if op == 'convert_numeric':
                if isQuantatative(df[column]):
                    if not ignoreWarnings:
                        warn(f'The conver_numeric option was set on {column}, which is not catagorical, skipping.')
                        continue
                else:
                    # match options:
                    if options == True or options == 'assign':
                        log(f'Converting "{column}" to quantatative by assinging to arbitrary values')
                        df[column], _ = pd.factorize(df[column])
                    elif options == 'one_hot_encode':
                        log(f'Converting "{column}" to quantatative by one hot encoding')
                        df = pd.get_dummies(df, columns=[column])
                    else:
                        raise TypeError(f"Bad arguement given to convert_numeric")
            if op == 'add_column':
                name, selection = options
                log(f'Adding new column "{name}"')
                # df[name] = df[column][selection]
                df[name] = selection
            if op == 'drop':
                if options:
                    log(f'Dropping column "{column}"')
                    df = df.drop(columns=[column])
    else:
        raise TypeError(f'Column "{column}" provided is not in the given DataFrame')

    return df

def clean(df:pd.DataFrame,
        config: Dict[str, Dict[str, Any]],
        verbose:bool=False,
    ) -> pd.DataFrame:
    """ Returns a cleaned copy of the DataFrame passed to it
        NOTE: The order of the entries in the config dict determine the order they are performed

        Arguments:
            config is a dict of this signature:
            NOTE: This is the suggested order
                {
                    # Do these to all the columns, or a specified column
                    'column/all': {
                        # Drop the column
                        'drop': bool,
                        # Drop duplicate samples
                        # Only applies to all
                        'drop_duplicates': bool,
                        # Maps feature values to a dictionary
                        'replace': Union[bool, Dict],
                        # Applies a function to the column
                        'apply': Union[bool, Callable],
                        # A list of (query, replacements)
                        'queries': Union[bool, List[Tuple[str, Union['remove', 'mean', 'median', 'mode', 'random', 'balanced_random', Any]]]]
                        # A ndarray of shape (1, n) of values to create a new column with the given name
                        # Calling from a specific column has no effect, behaves the same under all
                        'add_column': Tuple[str, np.ndarray],
                        # Specifies a value that is equivalent to the feature being missing
                        'missing_value': Any,
                        # Specifies a method by which to transform samples with missing features
                        # 'random' replaces missing values with either a random catagory, or a random number between min and max
                        # 'balanced_random' replaces missing values with either a randomly sampled catagory (sampled from the column
                        # itself, so it's properly biased), or a normally distributed sample
                        'handle_missing': Union[bool, 'remove', 'mean', 'median', 'mode', 'random', 'balanced_random', Any],
                        # Removes all samples with the given value
                        'remove': Union[bool, Any],
                        # Specifies a method by which to bin the quantative value, or specify custom ranges
                        'bin': Union[bool, Tuple['frequency', int], Tuple['width', int], Iterable],
                        # Specifies a method by which to normalize the quantative values
                        'normalize': Union[bool, 'min-max', 'range'],
                        # Specifies a method by which to convert a catagorical feature to a quantative one
                        'convert_numeric': Union[bool, 'assign', 'one_hot_encode'],
                },
    }
    """
    df = df.copy()
    log = lambda s: print(s) if verbose else None

    # Make sure the all section is done last (so if we're doing one hot encoding it doesn't throw errors)
    config = OrderedDict(config)
    if 'all' in config.keys():
        config.move_to_end('all')

    for column, args in config.items():
        log(f'Working on "{column}"')
        if column.lower() == 'all':
            # dropping duplicates means something different on the scale of a single column
            # than it does applied to the whole table
            if 'drop_duplicates' in args:
                log('Dropping duplicate samples')
                df = df.drop_duplicates()
                del args['drop_duplicates']

            for c in df.columns:
                # This makes a new args for a specific column, and removes any operations we've
                # already done (we want column specific options to override all, and we don't want
                # to redo them)
                adjusted = args.copy()
                if c in config.keys():
                    for op, params in config[c].items():
                        log(f'\tExcluding column {c} from {op}')
                        if op in adjusted.keys():
                            del adjusted[op]
                df = _cleanColumn(df, adjusted, c, verbose, True)
        else:
            df = _cleanColumn(df, args, column, verbose)
    return df

def resample(X, y, method:Union['oversample', 'undersample', 'mixed']='oversample', seed=None):
    # match method:
    if method == 'oversample':
        sampler = RandomOverSampler(random_state=seed)
        X, y = sampler.fit_resample(X, y)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=seed)
        X, y = sampler.fit_resample(X, y)
    elif method == 'mixed':
        todo('figure out how to mix under and over sampling')
    else:
        raise TypeError(f"Invalid method arguement given")
    return X, y

def ensemble(modelFunc, amt):
    """ Trains an {amt} of models for you, and then aggregates answers"""
    todo('Write ensemble function')

def fullTest(test, testPredictions, train=None, trainPredictions=None, accuracy=3, curve=False, confusion=True, explanation=False):
    assert (train is None) == (trainPredictions is None), 'You have to pass both train & trainPredictions'
    explain = lambda s: print('\t\t' + s) if explanation else None
    if isinstance(testPredictions[0].dtype, _catagoricalTypes):
        print('Test:')
        print(f'\tF1:        {sk.metrics.f1_score(test,        testPredictions):.{accuracy}}')
        explain('F1 is essentially an averaged score combining precision and recall')
        print(f'\tAccuracy:  {sk.metrics.accuracy_score(test,  testPredictions):.{accuracy}}')
        explain('Accuracy is a measure of how well the model did on average')
        print(f'\tPrecision: {sk.metrics.precision_score(test, testPredictions):.{accuracy}}')
        explain('Precision is a measure of how many things we said were true and we were wrong')
        print(f'\tRecall:    {sk.metrics.recall_score(test,    testPredictions):.{accuracy}}')
        explain('Recall is a measure of how many things we missed out on')
        if confusion:
            ConfusionMatrixDisplay.from_predictions(test, testPredictions, cmap='Blues')
        if curve:
            PrecisionRecallDisplay.from_predictions(test, testPredictions)
        plt.show()

        if train is not None and trainPredictions is not None:
            print('Train:')
            print(f'\tF1:        {sk.metrics.f1_score(train,        trainPredictions):.{accuracy}}')
            explain('F1 is essentially an averaged score combining precision and recall')
            print(f'\tAccuracy:  {sk.metrics.accuracy_score(train,  trainPredictions):.{accuracy}}')
            explain('Accuracy is a measure of how well the model did on average')
            print(f'\tPrecision: {sk.metrics.precision_score(train, trainPredictions):.{accuracy}}')
            explain('Precision is a measure of how many samples we accurately predicted?')
            print(f'\tRecall:    {sk.metrics.recall_score(train,    trainPredictions):.{accuracy}}')
            explain('Recall is a measure of how many times we accurately predicted a specific condition')
        if confusion:
            ConfusionMatrixDisplay.from_predictions(train, trainPredictions, cmap='Blues')
        if curve:
            PrecisionRecallDisplay.from_predictions(train, trainPredictions)
        plt.show()

    # Quantative measures
    else:
        print('Test:')
        # print(f'\tMean Square Error:      {mean_squared_error(test,      testPredictions):.{accuracy}}')
        print(f'\tRoot Mean Square Error: {sqrt(mean_squared_error(test, testPredictions)):.{accuracy}}')
        explain('An average of how far off we are from the target, in the same units as the target. Smaller is better.')
        print(f'\tMean Absolute Error:    {mean_absolute_error(test,     testPredictions):.{accuracy}}')
        explain('Similar to Root Mean Square Error, but better at weeding out outliers. Smaller is better.')
        print(f'\tR^2 Score:              {r2_score(test,                testPredictions):.{accuracy}}')
        explain('An average of how far off we are from just using the mean as a prediction. Larger is better.')

        if train is not None and trainPredictions is not None:
            print('Test:')
            # print(f'\tMean Square Error:      {mean_squared_error(train,      trainPredictions):.{accuracy}}')
            print(f'\tRoot Mean Square Error: {sqrt(mean_squared_error(train, trainPredictions)):.{accuracy}}')
            explain('An average of how far off we are from the target, in the same units as the target. Smaller is better.')
            print(f'\tMean Absolute Error:    {mean_absolute_error(train,     trainPredictions):.{accuracy}}')
            explain('Similar to Root Mean Square Error, but better at weeding out outliers. Smaller is better.')
            print(f'\tR^2 Score:              {r2_score(trian,                trainPredictions):.{accuracy}}')
            explain('An average of how far off we are from just using the mean as a prediction. Larger is better.')
