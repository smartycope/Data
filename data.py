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
from math import log, e

# If there's mutliple modes, how do we want to choose one? Used in _cleanColumn
# options: 'random', 'first', 'last'
MODE_SELECTION = 'random'

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

# np.object_ is a string type (apparently)
_catagoricalTypes = (str, Enum, np.object_, pd.CategoricalDtype, pd.Interval, pd.IntervalDtype, type(np.dtype('O')))

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

# This works, but it's slow for some reason?
def column_entropy(column:pd.Series, base=e):
    vc = pd.Series(column).value_counts(normalize=True, sort=False)
    return -(vc * np.log(vc)/np.log(base)).sum()

def pretty_2_column_array(a, ):
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

def quickSummary(data,
                 relevant=None,
                 target=None,
                 notRelevant=None,
                 stats=None,
                 additionalStats=[],
                 missing=True,
                 corr=.5,
                 entropy=None,
                 start='Description'
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
                'Head',
                'Stats',
                'Missing',
                'Duplicates',
                'Entropy',
                'Counts',
                'Correlations',
                'Features',
                'General Plots',
                'Specific Plots',
                'Matrix',
            ],
            value=start,
            description='Select Summary',
            # title='hello there'
        )

    # All the actual logic
    def output(x):
        # See baffled comment above
        corr, missing = whatTheHeck

        match x:
            case 'Description':
                print(f'There are {len(data)} samples, with {len(data.columns)} columns:')
                display(getNiceTypesTable(data))

                print()

                print('The possible values for the Catagorical values:')
                # This is jsut a complicated way to print them all nicely
                for key, value in sort_dict_by_value_length(dict([(c, data[c].unique()) for c in cat])).items():
                    print(key + ":")
                    joined_list = ", ".join(value)
                    if len(joined_list) <= 80: # adjust this number as needed
                        print('   ' + joined_list)
                    else:
                        for item in value:
                            print('   ' + item)
            case 'Stats':
                if len(quant):
                    # print('Summary of Quantatative Values:')
                    display(_relevant.agg(dict(zip(quant, [stats]*len(relevant)))))
            case 'Entropy':
                todo('Calculate entropy relative to the target feature')
                # if target is not None:
                # base = e if entropy is not None else entropy
                for c in data.columns:
                    print(f'The entropy of {c:>{max_name_len}} is: {round(_entropy(data[c].value_counts(normalize=True), base=entropy), 3)}')
                    # print(f'The entropy of {c} is: {entropy(data[c], data[target])}')
                # else:
                    # print('Target feature must be provided in order to calculate the entropy')
            case 'Duplicates':
                todo()
            case 'Head':
                display(data.head())
            case 'Counts':
                # This is sorted just so the features with less unique options go first
                for i in sorted(cat, key=lambda c: len(data[c].unique())):
                    print(f'{i} value counts:')

                    if len(data[i].unique()) == len(data[i]):
                        print('\tEvery sample has a unique catagory')
                    else:
                        print(pretty_counts(data[i]))
            case 'Correlations':
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
            case 'Missing':
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
            case 'Features':
                featureBox = widgets.Dropdown(
                    options=list(data.columns),
                    value=target,
                    description='Feature',
                )

                def showFeature(x):
                    group = 'catagorical' if isCatagorical(data[x]) else 'quantative'
                    type = data[x].dtype
                    # mode
                    missing = round(data[x].isnull().sum()/len(data[x]), 2)
                    display('hello!')

                    shared = f'{x} is a {group} feature of type {type}.\n' \
                             f'{missing:%} of it is NaN.\n' \
                             f'It has a mode of {mode}'

                    if isCatagorical(data[x]):
                        counts = pretty_counts(data[x])
                        this_entropy = round(_entropy(data[c].value_counts(normalize=True), base=entropy), 3)
                        print(shared +
                            f' and an entropy of {this_entropy}\n'
                            f'Value counts:\n'
                            + counts
                        )

                    # Quantative
                    else:
                        # std, mean, median
                        correlations = significantCorrelations(quant, corr)
                        print(shared +
                            f', an average value of {round(data[x].mean(), 2)}, and a median of {round(data[x].median(), 2)}'
                            f'It correlates with {correlations} by SOMETHING HERE'
                        )

                # print('test')
                # widgets.VBox([combobox, widgets.interactive(showFeature, x=featureBox)])
                display(featureBox)
                # display(widgets.VBox([combobox, featureBox]))
                # print('test2')

            case 'General Plots':
                if len(quant):
                    print('Plot of Quantatative Values:')
                    # if target in quantatative(data, relevant):
                    #     sns.catplot(data=quant, hue=target)
                    # else:
                    sns.catplot(data=quant)

                    plt.show()
                if len(cat):
                    print('Plot of Catagorical Value Counts:')
                    todo('catagorical (count?) plots')
                    # sns.categorical(data=cat)
                    # for d in cat.columns:
                    #     sns.categorical.countplot(data=cat[d])
                    #     # try:
                    #     if target in quantatative(data, relevant):
                    #         sns.countplot(data=cat[d], x=target)
                    #         sns.categorical(data=cat)
                    #         # sns.countplot(data=cat[d], hue=target)
                    #     else:
                    #         sns.countplot(data=cat[d])
                        # except ValueError:
                            # todo('Plot is still throwing annoying errors')
                    plt.show()
            case 'Specific Plots':
                if len(quant):
                    print('Plots of Quantatative Values:')
                    todo()

                if len(cat):
                    print('Plots of Catagorical Value Counts:')
                    todo()
            case 'Matrix':
                if len(quant):
                    print('Something Something Matrix:')
                    if target in quantatative(data, relevant):
                        sns.pairplot(data=quant, hue=target)
                    else:
                        sns.pairplot(data=quant)
                    plt.show()
                if len(cat) and False:
                    print('Box Plots of Catagorical Values:')
                    # sns.boxplot(data=cat, hue=target)
                    # display(cat)
                    # sns.relplot(data=cat, x='mfr',  y='name', hue=target)
                    plt.show()
            case _:
                print('Invalid start option')

    return widgets.interactive(output, x=combobox)

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
                match options:
                    case True:
                        if not ignoreWarnings:
                            raise TypeError(f"Please specify a value or method for the handle_missing option")
                    case False:
                        pass
                    case 'remove':
                        log(f'Removing all samples with a "{column}" values of "{missing}"')
                        df = without
                    case 'mean':
                        if isCatagorical(df[column]):
                            if not ignoreWarnings:
                                raise TypeError(f"Cannot get mean of a catagorical feature")
                            else:
                                continue
                        mean = without[column].mean()
                        log(f'Setting all samples with a "{column}" value of "{missing}" to the mean ({mean:.3})')
                        df.loc[df[column] == missing, column] = mean
                    case 'median':
                        if isCatagorical(df[column]):
                            if not ignoreWarnings:
                                raise TypeError(f"Cannot get median of a catagorical feature")
                            else:
                                continue
                        median = without[column].median()
                        log(f'Setting all samples with a "{column}" value of "{missing}" to the median ({median})')
                        df.loc[df[column] == missing, column] = median
                    case 'mode':
                        # I'm not sure how else to pick a mode, so just pick one at random
                        if MODE_SELECTION == 'random':
                            mode = random.choice(without[column].mode())
                        elif MODE_SELECTION == 'first':
                            mode = without[column].mode()[0]
                        elif MODE_SELECTION == 'last':
                            mode = without[column].mode()[-1]
                        log(f'Setting all samples with a "{column}" value of "{missing}" to a mode ({mode})')
                        df.loc[df[column] == missing, column] = mode
                    case _:
                        log(f'Setting all samples with a "{column}" value of "{missing}" to {options}')
                        df.loc[df[column] == missing, column] = options
            if op == 'remove':
                log(f'Removing all samples with a "{column}" value of {options}')
                df = df.loc[df[column] != options]
            if op == 'bin':
                if isCatagorical(df[column]):
                    if not ignoreWarnings:
                        warn(f'The bin option was set on "{column}", which is not quantatative, skipping.')
                        continue
                else:
                    match options:
                        case True:
                            if not ignoreWarnings:
                                raise TypeError(f"Please specify a method for the bin option")
                        case ('frequency', int()):
                            log(f'Binning "{column}" by frequency into {options[1]} bins')
                            df[column] = pd.qcut(df[column], options[1], duplicates='drop')
                        case ('width', int()):
                            log(f'Binning "{column}" by width into {options[1]} bins')
                            raise NotImplementedError('Width binning')
                        case tuple() | list():
                            log(f'Custom binning "{column}" into {len(options)} bins')
                            df[column] = pd.cut(df[column], options)
            if op == 'normalize':
                if isCatagorical(df[column]):
                    if not ignoreWarnings:
                        warn(f'The normalize option was set on {column}, which is not quantatative, skipping.')
                        continue
                else:
                    match options:
                        case True | 'min-max':
                            log(f'Normalizing "{column}" by min-max method')
                            df[column] = (df[column]-df[column].min())/(df[column].max()-df[column].min())
                        case 'range':
                            log(f'Normalizing "{column}" by range method')
                            raise NotImplementedError(f'range normalization doesn\'t work yet')
                            df[column] = (df[column]-df[column].mean())/df[column].std()
            if op == 'convert_numeric':
                if isQuantatative(df[column]):
                    if not ignoreWarnings:
                        warn(f'The conver_numeric option was set on {column}, which is not catagorical, skipping.')
                        continue
                else:
                    match options:
                        case True | 'assign':
                            log(f'Converting "{column}" to quantatative by assinging to arbitrary values')
                            df[column], _ = pd.factorize(df[column])
                        case 'one_hot_encode':
                            log(f'Converting "{column}" to quantatative by one hot encoding')
                            df = pd.get_dummies(df, columns=[column])
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
        config               :Dict[str, Dict[str, Any]],
        verbose              :bool=False,
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
                        # If provided, maps feature values to a dictionary
                        'replace': Union[bool, Dict],
                        # If provided, applies a function to the column
                        'apply': Union[bool, Callable],
                        # A ndarray of shape (1, n) of values to create a new column with the given name
                        # Calling from a specific column has no effect, behaves the same under all
                        'add_column': Tuple[str, np.ndarray],
                        # If provided, specifies a value that is equivalent to the feature being missing
                        'missing_value': Any,
                        # If provided, specifies a method by which to transform samples with missing features
                        'handle_missing': Union[bool, 'remove', 'mean', 'median', 'mode', Any],
                        # If provided, removes all samples with the given value
                        'remove': Union[bool, Any],
                        # If provided, specifies a method by which to bin the quantative value, or specify custom ranges
                        'bin': Union[bool, Tuple['frequency', int], Tuple['width', int], Iterable],
                        # If provided, specifies a method by which to normalize the quantative values
                        'normalize': Union[bool, 'min-max', 'range'],
                        # If provided, specifies a method by which to convert a catagorical feature to a quantative one
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
                # print(f'checking if {c} is in {config.keys()}')
                if c in config.keys():
                    for op, params in config[c].items():
                        # print(f'checking if {params} is False')
                        # if params == False:
                            # log(f'\tExcluding column {c} from {op}')
                        if op in adjusted.keys():
                            del adjusted[op]
                df = _cleanColumn(df, adjusted, c, verbose, True)
        else:
            df = _cleanColumn(df, args, column, verbose)
    return df

def resample(X, y, method:Union['oversample', 'undersample', 'mixed']='oversample', seed=None):
    match method:
        case 'oversample':
            sampler = RandomOverSampler(random_state=seed)
            X, y = sampler.fit_resample(X, y)
        case 'undersample':
            sampler = RandomUnderSampler(random_state=seed)
            X, y = sampler.fit_resample(X, y)
        case 'mixed':
            todo('figure out how to mix under and over sampling')
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
        explain('Precision is a measure of how many samples we accurately predicted?')
        print(f'\tRecall:    {sk.metrics.recall_score(test,    testPredictions):.{accuracy}}')
        explain('Recall is a measure of how many times we accurately predicted a specific condition')
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
