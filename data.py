# If y'all wanna contibute to my quickSummary function, that would be awesome.I'm trying
# to add more and more to it to make it even more useful. Here's the current todos I have:

# REMEMBER: Any edits you make in the file will NOT BE SAVED OR SHARED. If you make ANY EDITS
# please move them to the notebook (maybe we can make a section at the bottom)

## TODO
# - make the unique section be formatted better
# - add relplot with hue parameter
# - add a generalized catagorical graph
# - add hue to matrix graph
# - if there are no catagorical values, remove the unique print label
# - Fix all the commended out parts
# - Convert all the print statements to be graph titles instead
# - In summary, add info on the amounts of unique values of the target (for resampling)



# Here's some code you can use to test it
# from data import *
# data = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/cereal.csv')
# relevant = ('mfr', 'fat', 'calories', 'name', 'rating')
# target = 'rating'
# quickSummary(data,
#     relevant=relevant,
#     unique=False,
#     head=False,
#     stats=False,
#     corr=False,
#     missing=False,
#     plot=False,
#     plotAll=True,
#     matrix=False,
#     target='calories',
#     types=False,
# )



import pandas as pd
import seaborn as sns
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay
import ipywidgets as widgets
from typing import Union, Callable, Iterable
from sympy import Integer, Float

try:
    from Cope import todo
except ImportError:
    todo = lambda *a: print('TODO: ', *a)

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

def getNiceTypesTable(df, types=True):
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
    return pd.DataFrame(dict(zip(df.columns, [(['C'] if isinstance(df[i].dtype, _catagoricalTypes) else ['Q']) for i in df.columns])))

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

def quickSummary(data,
                 relevant=None,
                 target=None,
                 notRelevant=None,
                 stats=None,
                 additionalStats=[],
                 head=True,
                 #  describe=True,
                #  entropy=True,
                #  plot=True,
                #  plotAll=False,
                #  unique=True,
                #  matrix=True,
                 missing=60,
                 corr=.5,
                #  types=False,
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

    print(corr)

    # Define variables
    use = data[relevant]
    quant = data[quantatative(data, relevant)]
    cat = data[catagorical(data, relevant)]
    # Just to make sure we're catching all the columns
    # Convert to set so order doesn't matter
    assert set(quantatative(data, relevant) + catagorical(data, relevant)) == set(relevant)

    #define widgets
    combobox = widgets.Dropdown(
            options=[
                'Description',
                'Stats',
                'Entropy',
                'Duplicates',
                'Head',
                'Correlations',
                'Missing',
                'General Plots',
                'Specific Plots',
                'Matrix',
            ],
            value='Correlations',
            description='Select Summary',
            # title='hello there'
        )

    def output(x):
        if x == 'Description':
            print(f'There are {len(data)} samples, with {len(data.columns)} columns:')
            display(getNiceTypesTable(data, types=types))

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

        elif x == 'Stats':
            if len(quant):
                # print('Summary of Quantatative Values:')
                display(use.agg(dict(zip(quant, [stats]*len(relevant)))))

        elif x == 'Entropy':
            if target is not None:
                for c in data:
                    print(f'The entropy of {c} is: {entropy(data[c], target)}')
            else:
                print('Target feature must be provided in order to calculate the entropy')

        elif x == 'Duplicates':
            todo()

        elif x == 'Head':
            display(data.head())

        elif x == 'Correlations':
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

        elif x == 'Missing':
            if len(use):
                print('Missing Percentages:')
                if type(missing) is bool:
                    percent = use.isnull().sum()/len(use)*100
                    # This works, but instead I decided to overcomplicate it just so I can indent it
                    # print(percent)
                    offset = max(list(percent.index), key=len)
                    for i in range(len(percent)):
                        print(f'\t{percent.index[i]:>{len(offset)}}: {percent[i]}')
                elif isinstance(missing, (int, float)):
                    missing = normalizePercentage(missing)
                    _missing = missingSummary(use, missing/100)
                    if len(_missing):
                        display(_missing)
                    else:
                        print(f'\tAll values are missing less than {missing:.0%} of their entries')
                else:
                    raise TypeError('Missing is a bad type')

        elif x == 'General Plots':
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
                #         debug(1)
                #         sns.countplot(data=cat[d], x=target)
                #         sns.categorical(data=cat)
                #         debug(2)
                #         # sns.countplot(data=cat[d], hue=target)
                #     else:
                #         sns.countplot(data=cat[d])
                    # except ValueError:
                        # todo('Plot is still throwing annoying errors')
                plt.show()

        elif x == 'Specific Plots':
            if len(quant):
                print('Plots of Quantatative Values:')
                todo()

            if len(cat):
                print('Plots of Catagorical Value Counts:')
                todo()

        elif x == 'Matrix':
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

        else:
            print('Something horribly wrong has happened')

    return widgets.interactive(output, x=combobox)






def fullTest(test, testPredictions, train=None, trainPredictions=None, accuracy=3, curve=False, confusion=True):
    print(f'''
    Test:
        Accuracy:  {round(sk.metrics.accuracy_score(test,  testPredictions), accuracy)}
        F1:        {round(sk.metrics.f1_score(test,        testPredictions), accuracy)}
        Precision: {round(sk.metrics.precision_score(test, testPredictions), accuracy)}
        Recall:    {round(sk.metrics.recall_score(test,    testPredictions), accuracy)}
    ''')
    if confusion:
        ConfusionMatrixDisplay.from_predictions(test, testPredictions, cmap='Blues')
    if curve:
        PrecisionRecallDisplay.from_predictions(test, testPredictions)
    plt.show()

    assert (train is None) == (trainPredictions is None), 'You have to pass both train & trainPredictions '
    if train is not None and trainPredictions is not None:
        print(f'''
        Train:
            Accuracy:  {round(sk.metrics.accuracy_score(train,  trainPredictions), accuracy)}
            F1:        {round(sk.metrics.f1_score(train,        trainPredictions), accuracy)}
            Precision: {round(sk.metrics.precision_score(train, trainPredictions), accuracy)}
            Recall:    {round(sk.metrics.recall_score(train,    trainPredictions), accuracy)}
        ''')
    if confusion:
        ConfusionMatrixDisplay.from_predictions(train, trainPredictions, cmap='Blues')
    if curve:
        PrecisionRecallDisplay.from_predictions(train, trainPredictions)
    plt.show()





def normalize(df, method='default'):
    for col in quantatative(df):
        if method == 'default':
            df[col] = (df[col]-df[col].mean())/df[col].std()
        elif method == 'min-max':
            # display(df[col])
            df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())
        else:
            raise TypeError('Invalid method parameter')
_normalize = normalize

from typing import Optional, Any, Tuple, List, Iterable, Dict, Union



from Cope import debug

# TODO: add an option to handle unknowns by randomly insterting random values
# And another option to insert random values biased by how they are distributed in the data

def _cleanColumn(df, args, column, verbose, ignoreWarnings=False):
    debug(f'_cleanColumn called with {column}')
    log = lambda s: print(s) if verbose else None
    missing = np.nan
    if column in df.columns:
        for op, options in args.items():
            if op == 'drop_duplicates':
                if options == True:
                    log(f'Dropping duplicates in {column}')
                    df[column].drop_duplicates(inplace=True)
            if op == 'map':
                if options == True:
                    if not ignoreWarnings:
                        raise TypeError(f"Please specify a dict for the map option")
                if isinstance(options, dict):
                    log(f'Mapping {column}')
                    df[column].map(options)
            if op == 'apply':
                if options == True:
                    if not ignoreWarnings:
                        raise TypeError(f"Please specify a function to apply")
                elif callable(options):
                    log(f'Applying function to {column}')
                    df[column].apply(options)
            if op == 'missing_value':
                missing = options
            if op == 'handle_missing':
                without = data[column][data[column] != missing]
                match options:
                    case True:
                        if not ignoreWarnings:
                            raise TypeError(f"Please specify a value or method for the handle_missing option")
                    case False:
                        pass
                    case 'remove':
                        log(f'Removing all samples with a {column} values of {missing}')
                        df[column] = without # Note
                    case 'mean':
                        if isCatagorical(df[column]):
                            if not ignoreWarnings:
                                raise TypeError(f"Cannot get mean of a catagorical feature")
                        mean = without.mean()
                        log(f'Setting all samples with a {column} value of {missing} to the mean ({mean})')
                        df[column].loc[df[column] == missing] = mean
                    case 'median':
                        if isCatagorical(df[column]):
                            if not ignoreWarnings:
                                raise TypeError(f"Cannot get median of a catagorical feature")
                        median = without.median()
                        log(f'Setting all samples with a {column} value of {missing} to the median ({median})')
                        df[column].loc[df[column] == missing] = median
                    case 'mode':
                        mode = without.mode()
                        log(f'Setting all samples with a {column} value of {missing} to the mode ({mode})')
                        df[column].loc[df[column] == missing] = mode
                    case _:
                        log(f'Setting all samples with a {column} value of {missing} to {options}')
                        df[column].loc[df[column] == missing] = options
            if op == 'remove':
                log(f"Removing all samples with a {column} value of {options}")
                df[column] = df[column][df[column] != options] # Note
            if op == 'bin':
                if isCatagorical(df[column]):
                    if not ignoreWarnings:
                        raise Warning(f'The bin option was set on {column}, which is not quantatative, skipping.')
                else:
                    match options:
                        case True:
                            if not ignoreWarnings:
                                raise TypeError(f"Please specify a method for the bin option")
                        case ('frequency', int()):
                            log(f'Binning {column} by frequency into {options[1]} bins')
                            df[column] = pd.qcut(df[column], options[1])
                        case ('width', int()):
                            log(f'Binning {column} by width into {options[1]} bins')
                            raise NotImplementedError('Width binning')
                        case tuple() | list():
                            log(f'Custom binning {column} into {len(options)} bins')
                            df[column] = pd.cut(df[column], options)
            # display(df)
            # display(df['name'])
            # print(f'|{column}|')
            # display(df[column])
            if op == 'normalize':
                if isCatagorical(df[column]):
                    if not ignoreWarnings:
                        raise Warning(f'The normalize option was set on {column}, which is not quantatative, skipping.')
                else:
                    match options:
                        case True | 'range':
                            log(f'Normalizing {column} by range method')
                            raise NotImplementedError(f'range normalization doesn\'t work yet')
                            df[column] = (df[column]-df[column].mean())/df[column].std()
                        case 'min-max':
                            log(f'Normalizing {column} by min-max method')
                            df[column] = (df[column]-df[column].min())/(df[column].max()-df[column].min())
            if op == 'convert_numeric':
                if isQuantatative(df[column]):
                    if not ignoreWarnings:
                        raise Warning(f'The conver_numeric option was set on {column}, which is not catagorical, skipping.')
                else:
                    match options:
                        case True | 'assign':
                            log(f'Converting {column} to quantatative by assinging arbitrary values')
                            df[column], _ = pd.factorize(df[column])
                        case 'one_hot_encode':
                            log(f'Converting {column} to quantatative by one hot encoding')
                            df = pd.get_dummies(df, columns=[column])
            if op == 'create_new':
                name, selection = options
                log(f'Creating new column {name} from {column}')
                df[name] = df[column][selection]
            if op == 'drop':
                log(f'Dropping column {column}')
                df = df.drop(columns=drop)
    else:
        raise TypeError(f"Column {column} provided is not in the given DataFrame")


def clean(df:pd.DataFrame,
        # target               :str,
        config               :Dict[str, Dict[str, Any]],
        # resampling           :Union[bool, 'oversample', 'undersample', 'mixed']=False,
        verbose              :bool=False,
        # inplace=False,
    ) -> pd.DataFrame:
    """ Returns a cleaned copy of the DataFrame passed to it

        NOTE: The order of the entries in the config dict determine the order they are performed

        Arguments:
            config is a dict of this signature:
                {
                    # Do these to all the columns, or a specified column
                    'column/all': {
                        # Drop the column
                        'drop': bool,
                        # Drop duplicate samples
                        'drop_duplicates': bool,
                        # If provided, maps feature values to a dictionary
                        'map': Union[bool, Dict],
                        # If provided, applies a function to the column
                        'apply': Union[bool, Callable],
                        # If provided, specifies a value that is equivalent to the feature being missing
                        'missing_value': Any,
                        # If provided, specifies a method by which to transform samples with missing features
                        'handle_missing': Union[bool, 'remove', 'mean', 'median', 'mode', Any],
                        # If provided, specifies a method by which to bin the quantative value, or specify custom ranges
                        'bin': Union[bool, Tuple['frequency', int], Tuple['width', int], Iterable],
                        # If provided, removes all samples with the given value
                        'remove': Union[bool, Any],
                        # A ndarray of shape (1, n) of bools to apply to the column to create a new column with the given name
                        # Usable on single columns only (not all)
                        'create_new': Tuple[str, np.ndarray],
                        # If provided, specifies a method by which to normalize the quantative values
                        'normalize': Union[bool, 'min-max', 'range'],
                        # If provided, specifies a method by which to convert a catagorical feature to a quantative one
                        'convert_numeric': Union[bool, 'assign', 'one_hot_encode'],
                },
    }
    """
    df = df.copy()
    log = lambda s: print(s) if verbose else None

    for column, args in config.items():
        if column.lower() == 'all':
            # dropping duplicates means something different on the scale of a single column
            # than it does applied to the whole table
            if 'drop_duplicates' in args:
                df.drop_duplicates()
                del args['drop_duplicates']

            for c in df.columns:
                if c in config:
                    # We want to keep all the custom configs, and add any unspecified ones that
                    # are specified by all to them
                    new = args.copy()
                    new.update(config[c])
                    print(f'args updated with {c}')
                    display(args)
                    _cleanColumn(df, new, c, verbose, True)
                else:
                    print(f'args in all loop doing {c}')
                    display(args)
                    _cleanColumn(df, args, c, verbose, True)
        else:
            print(f'args: {column}')
            display(args)
            _cleanColumn(df, args, column, verbose)
        return df






























"""







    # Age
    if 'age' in df.columns:
        if AGE_BINNING_METHOD == 'frequency':
            df.age = pd.qcut(df.age, AGE_BIN_AMT)
        elif AGE_BINNING_METHOD == 'width':
            todo("age width binnig")
        elif AGE_BINNING_METHOD == 'custom':
            df.age = pd.cut(df['age'], [17, 25, 40, 60, 75, 200])
        else:
            todo('normalize age')
    # Job
    if 'job' in df.columns:
        if REMOVE_JOB_STUDENT:
            df.job = df.job[df.job != 'student']
        if REMOVE_JOB_UNEMPLOYED:
            df.job = df.job[df.job != 'unemployed']
    # Marital
    if 'marital' in df.columns:
        if REMOVE_MARITAL_UNKNOWN:
            df.marital = df.marital[df.marital != 'unknown']
    # Education
    if 'education' in df.columns:
        if QUANTIFY_EDUCATION:
            df.education = df.education.map({
                'basic.4y': 4,
                'high.school': 9,
                'basic.6y': 6,
                'basic.9y': 9,
                'professional.course': 10,
                'unknown': 9, # assume they at least passed highschool
                'university.degree': 12,
                'illiterate': 0,
            })
    # Default
    if 'default' in df.columns:
        if REMOVE_DEFAULTED:
            df.default = df.default[df.default != 'yes']

        if DEFUALT_UNKNOWN_METHOD == 'remove':
            df.default = df.default[df.default != 'unknown']
        elif DEFUALT_UNKNOWN_METHOD == 'yes':
            df.default.loc[df.default == 'unknown'] = 'yes'
        elif DEFUALT_UNKNOWN_METHOD == 'mode':
            df.default.loc[df.default == 'unknown'] = df.default.mode()[0]
    # Housing
    if 'housing' in df.columns:
        if HOUSING_UNKNOWN_METHOD == 'remove':
            df.housing = df.housing[df.housing != 'unknown']
        elif HOUSING_UNKNOWN_METHOD == 'mode':
            df.housing.loc[df.housing == 'unknown'] = df.housing.mode()[0]
    # Loan
    if 'loan' in df.columns:
        if LOAN_UNKNOWN_METHOD == 'remove':
            df.loan = df.loan[df.loan != 'unknown']
        if LOAN_UNKNOWN_METHOD == 'mode':
            df.housing.loc[df.loan == 'unknown'] = df.loan.mode()[0]
    # pdays
    if 'pdays' in df.columns:
        without = data.pdays[data.pdays != 999]
        if PDAYS_999_METHOD == 'remove':
            df.pdays = without
        if PDAYS_999_METHOD == 'average':
            df.pdays.loc[df.pdays == 999] = without.mean()
        if PDAYS_999_METHOD == '0':
            df.pdays.loc[df.pdays == 999] = 0
        if PDAYS_999_METHOD == 'Add was_contacted column':
            df['was_contacted'] = df.pdays[df.pdays != 999]
            df.was_contacted.loc[df.was_contacted.isnull()] = 0
            df.was_contacted.loc[df.was_contacted != 0.] = 1
            df.pdays.drop(columns='pdays')
            # display(df.was_contacted.unique())

    # Normalize the quantative columns to within a certain range
    if NORMALIZE:
        print('normalizing...')
        _normalize(df, method=NORMALIZE_METHOD)

    # Convert Catagorical data to unique integers
    print('converting y column to 1s and 0s...')
    if 'y' in df.columns:
        df.y = df.y.map({'yes': 1, 'no': 0})

    if NUMERIC_CONVERSION == 'assign':
        # CONVERT Y COLUMN FIRST
        # We have to have the yes's = 1 and the no's = 0, otherwise we're training the model backwards
        assignments = {}
        for c in catagorical(df):
            # print(f'converting catagorical column {c} to numerical...')
            df[c], assignments[c] = pd.factorize(df[c])
    elif NUMERIC_CONVERSION == 'oneHotEncode':
        df = pd.get_dummies(df)

    # Split the target column from the rest of the data if we have it
    if 'y' in df.columns:
        # This is just here so we dont drop duplicates of the holdout set
        if DROP_DUPLICATES:
            df.drop_duplicates(inplace=True)

        X, y = df.drop(columns='y'), df['y']

        # Resample the data only if we have the answers
        if RESAMPLING == 'oversample':
            sampler = RandomOverSampler(random_state=RESAMPLE_SEED)
            X, y = sampler.fit_resample(X, y)
        elif RESAMPLING == 'undersample':
            sampler = RandomUnderSampler(random_state=RESAMPLE_SEED)
            X, y = sampler.fit_resample(X, y)
        elif RESAMPLING == 'both':
            todo('figure out how to mix under and over sampling')

        return X, y

    # otherwise, we're just dfing the holdout set
    else:
        return df

 """

#@title Cleaning Configs {display-mode: "form"}
#@markdown Cleaning Configuration

# CONVERT_TO_NUMERIC = True #@param ["True", "False"] {type:"raw"}
# NORMALIZE = True #@param ["True", "False"] {type:"raw"}
# NORMALIZE_METHOD = 'min-max' #@param ["default", "min-max"]
# ADD_WAS_CONTACTED_COLUMN = True #@param ["True", "False"] {type:"raw"}
# DROP_DUPLICATES = True #@param ["True", "False"] {type:"raw"}
AGE_BINNING_METHOD = 'frequency' #@param ['frequency', 'width', 'custom']
AGE_BIN_AMT = 5 #@param {type:"integer"}
REMOVE_JOB_STUDENT = True #@param ["True", "False"] {type:"raw"}
REMOVE_JOB_UNEMPLOYED = True #@param ["True", "False"] {type:"raw"}
REMOVE_MARITAL_UNKNOWN = False #@param ["True", "False"] {type:"raw"}
QUANTIFY_EDUCATION = True #@param ["True", "False"] {type:"raw"}
REMOVE_DEFAULTED = False #@param ["True", "False"] {type:"raw"}
# If you don't know, assume the worst
NUMERIC_CONVERSION = 'assign' #@param ['oneHotEncode', 'assign']
DEFUALT_UNKNOWN_METHOD = 'yes' #  @param ['remove', 'yes', 'mode']
HOUSING_UNKNOWN_METHOD = 'mode' # @param ['remove', 'mode']
LOAN_UNKNOWN_METHOD    = 'mode' # @param ['remove', 'mode']
PDAYS_999_METHOD = 'Add was_contacted column' # @param ['remove', 'average', '0', 'Add was_contacted column']
RESAMPLING = 'oversample' #@param ['oversample', 'undersample', 'both', 'neither']
# ONE_HOT_ENCODE = True #@param ["True", "False"] {type:"raw"}
SPLIT_SEED = 12345 #@param {type:"integer"}
RESAMPLE_SEED = 54321 #@param {type:"integer"}
REMOVE_COLUMNS = ['contact', 'nr.employed'] #@param
CLASS_WEIGHT_PARAM = "balanced" #@param {type:"raw"}
