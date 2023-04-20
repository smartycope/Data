import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from functools import wraps
from contextlib import redirect_stdout
from imblearn.under_sampling import RandomUnderSampler
from warnings import warn
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import sklearn
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
# from scipy.stats import entropy as _entropy
# from scipy.stats import kurtosis
import scipy.stats
import matplotlib.pyplot as plt
from typing import Optional, Any, Tuple, List, Iterable, Dict, Union, Callable, Iterable, Literal
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay
import ipywidgets as widgets
from collections import OrderedDict
from IPython.display import clear_output, display
from math import log, e
import sklearn.model_selection as skms

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
# This is an option for the zscore slider in explore(). Turning it on is fancier, but it's really slow
CONTINUOUS_UPDATE_SLIDER = False

# For use with DataFrame.select_dtypes(include=/exclude=)
# _catagoricalTypes = (str, Enum, np.object_, pd.CategoricalDtype, pd.Interval, pd.IntervalDtype, type(np.dtype('O')), bool, np.bool_, np.bool8)
_catagoricalTypes = ['bool', 'bool_', 'object', 'object_', 'Interval', 'bool8', 'category']
_quantitativeTypes = ['number']
_timeTypes = ['datetimetz', 'timedelta', 'datetime']


try:
    # from Cope import todo
    pass
except ImportError:
    todo = lambda *a: print('TODO: ', *a)


# I got tired of MinMaxScaler returning numpy arrays
def _cast2dataframe(func):
    def wrapper(self, *args, **kwargs):
        return pd.DataFrame(func(self, *args, **kwargs), columns=self.feature_names_in_)
    return wrapper

MinMaxScaler.transform = _cast2dataframe(MinMaxScaler.transform)

def installLibs(libs=['pandas', 'numpy', 'imblearn', 'ipywidgets', 'seaborn', 'scipy', 'matplotlib']):
    libs = ' '.join(libs)
    try:
        import IPython
    except:
        print(f'IPython doesnt seem to be installed. Simply run `pip install {libs}` in a terminal')

    if (ipython := IPython.get_ipython()) is not None:
        ipython.run_line_magic("pip", f"install {libs}")
    else:
        print('You dont seem to be calling from IPython. Simply run `pip install pandas altair numpy imblearn ipywidgets seaborn scipy matplotlib IPython` in a terminal')


# I AM THE COMPUTER GOBLIN
#       FEAR ME
def addVerbose(func):
    # Runs when the decorator is added
    @wraps(func)
    def inner(*args, verbose=False, **kwargs):
        # Runs when the decorated function gets called
        return func(
            *args,
            log=lambda s: print(f'\t{s}') if verbose else None,
            **kwargs)
    return inner

def _cleaning_func(**decorator_kwargs):
    """ Auto-converts the given named parameter to the given type
        Supports inputs of pd.DataFrame, pd.Series, np.ndarray, and tuple/list pd.Series or np.ndarray
        Supports outputs of pd.DataFrame, pd.Series, and a tuple of pd.Series
        Does NOT support input types of tuple/list of pd.DataFrames
    """
    trivial = lambda x: x

    def error(toType):
        def _error(x):
            raise TypeError(f"Cant cast {toType} to {type(x)}")
        return _error

    iterableInput = {
        pd.DataFrame: lambda t: pd.DataFrame(t).T,
        pd.Series:    lambda t: (pd.Series(t[0]) if len(t) == 1 else error(pd.Series)(t)),
        tuple:        lambda t: tuple([pd.Series(i) for i in t]),
    }

    input2output = {
        pd.DataFrame: {
            pd.DataFrame: trivial,
            pd.Series:    lambda d: pd.Series(d.iloc[:,0]) if len(d.columns) == 1 else error(pd.Series),
            tuple:        lambda d: tuple([d[i] for i in d]),
        },
        pd.Series:    {
            pd.DataFrame: lambda s: pd.DataFrame(s),
            pd.Series:    trivial,
            tuple:        lambda s: (s,),
        },
        np.ndarray:   {
            pd.DataFrame: lambda n: pd.DataFrame(n),
            pd.Series:    lambda n: pd.Series(n),
            tuple:        lambda s: (pd.Series(s),),
        },
        tuple: iterableInput,
        list:  iterableInput,
    }

    def outer(decorator_func):
        # Also runs when the decorator is added
        @wraps(decorator_func)
        @addVerbose
        def inner(dat, *args, **kwargs):
            # Runs when the decorated function gets called
            if isinstance(dat, (list, tuple)):
                if len(dat) == 0:
                    raise TypeError('Please dont pass in an empty list')
                elif len(dat) == 1:
                    dat = dat[0]
                # If we're given a collection of pd.DataFrames, then iterate through the function and
                # apply it to all of them
                elif isinstance(dat[0], pd.DataFrame):
                    _kwargs = kwargs.copy()
                    rtn = []
                    for d in dat:
                        for paramName, outputType in decorator_kwargs.items():
                            _kwargs[paramName] = input2output[pd.DataFrame][outputType](d)
                        rtn.append(decorator_func(*args, **kwargs))
                    return rtn

            for paramName, outputType in decorator_kwargs.items():
                kwargs[paramName] = input2output[type(dat)][outputType](dat)
            return decorator_func(*args, **kwargs)
        return inner
    return outer


def insertSample(df, sample, index=-1):
    """ Because theres not a function for this? """
    df.loc[index - .5] = sample
    return df.sort_index().reset_index(drop=True)

def ensureIterable(obj, useList=False):
    if not isiterable(obj):
        return [obj, ] if useList else (obj, )
    else:
        return obj

def ensureNotIterable(obj, emptyBecomes=None):
    if isiterable(obj):
        # Generators are iterable, but don't inherantly have a length
        try:
            len(obj)
        except:
            obj = list(obj)

        if len(obj) == 1:
            try:
                return obj[0]
            except TypeError:
                return list(obj)[0]
        elif len(obj) == 0:
            return obj if emptyBecomes is _None else emptyBecomes
        else:
            return obj
    else:
        return obj

def getOutliers(data, zscore=None):
    # TODO: add more options here (like getting outliers via kurtosis & IQR)
    # IQR (inner quartile range) = Q3-Q1
    # +/- 1.5*IQR == possible outlier
    # +/- 3*IQR == outlier
    # kurtosis
    if isinstance(data, pd.Series):
        if zscore is not None:
            return data[np.abs(scipy.stats.zscore(data)) > zscore]

    elif isinstance(data, pd.DataFrame):
        if zscore is not None:
            rtn = {}
            for f in data.columns:
                rtn[f] = data[f][np.abs(scipy.stats.zscore(data[f])) > zscore]
            return pd.DataFrame(rtn)

    else:
        raise TypeError(f"Invalid type {type(data)} given")

def normalizePercentage(p, error='Percentage is of the wrong type (int or float expected)'):
    if isinstance(p, int):
        return p / 100
    elif isinstance(p, float):
        return p
    elif isinstance(p, bool):
        if p is True:
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

def timeFeatures(df) -> pd.DataFrame:
    return df.select_dtypes(include=_timeTypes)

def catagorical(df, time=False) -> pd.DataFrame:
    return df.select_dtypes(include=_catagoricalTypes + (_timeTypes if time else []))

def quantitative(df, time=True) -> pd.DataFrame:
    return df.select_dtypes(include=_quantitativeTypes + (_timeTypes if time else []))

def isTimeFeature(s: pd.Series):
    s = pd.Series(s, name='__dummy')
    return s.name in timeFeatures(pd.DataFrame(s))

def isCatagorical(s: pd.Series, time=False):
    s = pd.Series(s, name='__dummy')
    return s.name in catagorical(pd.DataFrame(s), time)

def isQuantatative(s: pd.Series, time=True):
    s = pd.Series(s, name='__dummy')
    return s.name in quantitative(pd.DataFrame(s), time)

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
    def _getLabels(col):
        if isCatagorical(col, time=False):
            return [col.dtype, 'C']
        if isQuantatative(col, time=False):
            return [col.dtype, 'Q']
        if isTimeFeature(col):
            return [col.dtype, 'T']

    return pd.DataFrame(dict(zip(
        df.columns,
        [_getLabels(df[f]) for f in df.columns]
    )))

def percentCountPlot(data, feature, target=None, ax=None, title='Percentage of values used in {}'):
    # plt.figure(figsize=(20,10))
    # plt.title(f'Percentage of values used in {feature}')
    Y = data[feature]
    total = float(len(Y))
    ax = sns.countplot(x=feature, data=data, hue=target, ax=ax)
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

def pretty_2_column_array(a, limit=30, paren=None):
    card = len(a)
    if card > limit:
        a = a[:limit-1]
        # a.append(f'... ({card - limit - 1} more)')

    offset = max(list(a.index), key=len)
    rtn = ''
    for i in range(len(a)):
        if paren is None:
            rtn += f'\t{a.index[i]:>{len(offset)}}: {a[i]:.1%}\n'
        else:
            rtn += f'\t{a.index[i]:>{len(offset)}}: {a[i]:.1%} ({paren[i]})\n'
    return rtn

def pretty_counts(s:pd.Series, paren=False):
    # rtn = ''
    # for i in s.value_counts(normalize=True, sort=True):
    #     rtn += str(i)
    # rtn = str()
    if paren:
        rtn = pretty_2_column_array(s.value_counts(normalize=True, sort=True), paren=s.value_counts(sort=True))
    else:
        rtn = pretty_2_column_array(s.value_counts(normalize=True, sort=True))
    return rtn


def meanConfInterval(data, confidence=0.95, mean=False):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    if mean:
        return m, m-h, m+h
    else:
        return m-h, m+h

def showOutliers(data, column, zscore, **snsArgs):
    if isCatagorical(data[column]):
        raise TypeError('Outliers only apply to quantitative values')
    samples = getOutliers(data[column], zscore=zscore)
    print(len(samples), len(data[column]), sep='/')
    sns.scatterplot(data=data[column], **snsArgs)
    sns.scatterplot(data=samples, **snsArgs)
    plt.show()

def interactWithOutliers(df, feature=None, step=.2):
    return widgets.interactive(showOutliers,
        data=widgets.fixed(df),
        column=list(df.columns) if feature is None else widgets.fixed(feature),
        zscore=(0., df[feature].max() / df[feature].std(), step) if feature is not None else (0., 10, step)
        # zscore=(0., 20, step)
    )

# Clean Functions
@_cleaning_func(col=pd.Series)
def handle_outliers(col, method:Union['remove', 'constrain']='remove', zscore=3, log=...):
    # TODO: add more options here (like getting outliers via kurtosis & IQR)
    samples = getOutliers(col, zscore=zscore)
    if method == 'remove':
        log(f'Removing outliers with zscore magnitudes >{zscore} from {col.name}')
        return col.drop(samples.index)
    elif method == 'constrain':
        todo('This breaks on negative values')
        # todo try optionally getting everything *not* in range instead of just the things in range
        # The value that corresponds to a given score is the standard deviate * zscore
        max = col.std() * zscore
        # df.loc[samples.index, column] = np.clip(samples, -max, max)
        log(f'Constraining outliers with zscore magnitudes >{zscore} from {col.name}')
        # col[samples.index] = np.clip(samples, -max, max)
        # col.mask()
        return col.apply(lambda s: np.clip(s, -max, max))
    else:
        raise TypeError(f"Invalid method arguement '{method}' given")

@_cleaning_func(col=pd.Series)
def handle_missing(col, method:Union[pd.Series, 'remove', 'mean', 'median', 'mode', 'random', 'balanced_random', Any], missing_value=np.nan, log=...):
    without = col.loc[col != missing_value]
    # match options:
    if isinstance(method, (pd.Series, np.ndarray)):
        assert len(method) == len(col), 'Both arrays are not of the same length'
        log(f'Replacing all samples with a "{col.name}" value of "{missing_value}" with their indexes in "{method.name}"')
        # return col.apply(lambda sample: method[sample.index] if sample == missing_value else sample)
        return pd.Series([(method[i] if col[i] == missing_value else col[i]) for i in range(len(col))])

        # return pd.Series(col.reset_index().apply(lambda i: method[i] if col[i] == missing_value else col[i], axis=1).values, index=col.index)
        # return col.apply(lambda sample: method[sample.index] if sample == missing_value else sample)
    elif method == 'remove':
        log(f'Removing all samples with "{col.name}" values of "{missing_value}"')
        return without
    elif method == 'mean':
        if isCatagorical(col):
            raise TypeError("Cannot get mean of a catagorical feature")
        mean = without.mean()
        log(f'Setting all samples with a "{col.name}" value of "{missing_value}" to the mean ({mean:.2f})')
        # Copy it for consistency
        return col.copy().mask(col == missing_value, mean)
    elif method == 'median':
        if isCatagorical(col):
            raise TypeError("Cannot get median of a catagorical feature")
        median = without.median()
        log(f'Setting all samples with a "{col.name}" value of "{missing_value}" to the median ({median})')
        return col.copy().mask(col == missing_value, median)
    elif method == 'mode':
        # I'm not sure how else to pick a mode, so just pick one at random
        if MODE_SELECTION == 'random':
            mode = random.choice(without.mode())
        elif MODE_SELECTION == 'first':
            mode = without.mode()[0]
        elif MODE_SELECTION == 'last':
            mode = without.mode()[-1]
        log(f'Setting all samples with a "{col.name}" value of "{missing_value}" to a mode ({mode})')
        return col.copy().mask(col == missing_value, mode)
    elif method == 'random':
        if isCatagorical(col):
            log(f'Setting all samples with a "{col.name}" value of "{missing_value}" to random catagories')
            fill = lambda sample: random.choice(without.unique()) if sample == missing_value else sample
        else:
            log(f'Setting all samples with a "{col.name}" value of "{missing_value}" to random values along a uniform distrobution')
            fill = lambda sample: type(sample)(random.uniform(without.min(), without.max())) if sample == missing_value else sample

            return col.apply(fill)
    elif method == 'balanced_random':
        if isCatagorical(col):
            log(f'Setting all samples with a "{col.name}" value of "{missing_value}" to evenly distributed random catagories')
            fill = lambda sample: random.choice(without) if sample == missing_value else sample
        else:
            log(f'Setting all samples with a "{col.name}" value of "{missing_value}" to random values along a normal distrobution')
            fill = lambda sample: type(sample)(random.gauss(without.mean(), without.std())) if sample == missing_value else sample
        return col.apply(fill)
    else:
        log(f'Setting all samples with a "{col.name}" value of "{missing_value}" to {method}')
        # col.loc[col == missing_value] = method
        return col.copy().mask(col == missing_value, method)
    return col

def query(df:pd.DataFrame, column:str, query:str, method:Union[pd.Series, 'remove', 'new', 'mean', 'median', 'mode', 'random', 'balanced_random', Any], true=1, false=0, verbose=False):
    df = df.copy()
    if isinstance(method, pd.Series):
        log(f'Changing all samples where "{query}" is true to have the {column} values of their indecies in "{method.name}"')
        q = df.query(query)
        df.loc[q.index, column] = q.apply(lambda s: method[s.name], axis=1)
    elif method == 'remove':
        log(f'Removing all samples where "{query}" is true')
        df = df.drop(df.query(query).index)
    elif method == 'mean':
        if isCatagorical(df[column]):
            raise TypeError("Cannot get mean of a catagorical feature")
        mean = df[column].mean()
        log(f'Setting all samples where {query} is true to the mean of "{column}" ({mean:.2})')
        df.loc[df.query(query).index, column] = mean
    elif method == 'median':
        if isCatagorical(df[column]):
            raise TypeError("Cannot get median of a catagorical feature")
        median = df[column].median()
        log(f'Setting all samples where "{query}" is true to the median of "{column}" ({median})')
        df.loc[df.query(query).index, column] = median
    elif method == 'mode':
        # I'm not sure how else to pick a mode, so just pick one at random
        if MODE_SELECTION == 'random':
            mode = random.choice(df[column].mode())
        elif MODE_SELECTION == 'first':
            mode = df[column].mode()[0]
        elif MODE_SELECTION == 'last':
            mode = df[column].mode()[-1]
        log(f'Setting all samples where "{query}" is true to a mode of "{column}" ({mode})')
        df.loc[df.query(query).index, column] = mode
    elif method == 'random':
        if isCatagorical(df[column]):
            log(f'Setting all samples where "{query}" is true to have random catagories')
            fill = lambda s: random.choice(df[column].unique())
        else:
            log(f'Setting all samples where "{query}" is true to have random values along a uniform distrobution')
            fill = lambda s: type(s)(random.uniform(df[column].min(), df[column].max()))

        q = df.query(query)
        df.loc[q.index, column] = q[column].apply(fill)
    elif method == 'new':
        q = df.query(query)
        df[column] = false
        df.loc[q.index, column] = true
    elif method == 'balanced_random':
        if isCatagorical(df[column]):
            log(f'Setting all samples where "{query}" is true to have evenly distributed random catagories')
            fill = lambda s: random.choice(df[column])
        else:
            log(f'Setting all samples where "{query}" is true to have random values along a normal distrobution')
            fill = lambda s: type(s)(random.gauss(df[column].mean(), df[column].std()))

        q = df.query(query)
        df.loc[q.index, column] = q[column].apply(fill)
    else:
        log(f'Setting all samples where "{query}" is true to have a "{column}" value of  {method}')
        df.loc[df.query(query).index, column] = method
    return df

@_cleaning_func(col=pd.Series)
def remove(col, val, log=...):
    log(f'Removing all samples with a "{col.name}" value of {val}')
    # return col.mask(col == val, val)
    return col.drop(index=col[col == val].index)

@_cleaning_func(col=pd.Series)
def bin(col, method:Union['frequency', 'width', Tuple, List], amt=5, log=...):
    if isCatagorical(col):
        raise TypeError(f"Can't bin catagorical feature '{col.name}'")

    if   method == 'frequency':
        log(f'Binning "{col.name}" by frequency into {amt} bins')
        return pd.qcut(col, amt, duplicates='drop')
    elif method == 'width':
        log(f'Binning "{col.name}" by width into {amt} bins')
        raise NotImplementedError('Width binning')
    elif isinstance(method, (tuple, list)):
        log(f'Custom binning "{col.name}" into {len(method)} bins')
        return pd.cut(col, method)
    else:
        raise TypeError(f"Bin method parameter given invalid option {method}")

@_cleaning_func(df=pd.DataFrame)
def rescale(df, return_scaler=False, log=...):
    log('Rescaling')
    # display(df)
    scaler = MinMaxScaler().fit(df)
    ans = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return (ans, scaler) if return_scaler else ans

def convert_time(df_or_col, col:str=None, method:Union['timestamp']='timestamp', verbose=False):
    assert not (isinstance(df_or_col, pd.Series) and col is not None), 'Please dont provide a col parameter if passing a Series'
    if isinstance(df_or_col, pd.DataFrame) and col is None:
        df = df_or_col.copy()
        df[timeFeatures(df).columns] = timeFeatures(df).applymap(lambda date: date.timestamp())
        return df
    else:
        if isinstance(df_or_col, pd.DataFrame):
            df_or_col = df_or_col[col]
        return df_or_col.apply(lambda date: date.timestamp())

def convert_numeric(df, col:str=None, method:Union['assign', 'one_hot_encode']='one_hot_encode', returnAssignments=False, skip=[], verbose=False):
    df = df.copy()
    # if isinstance(df, pd.Series) and method == 'one_hot_encode':
        # raise TypeError("A DataFrame and column name is required when using one hot encoding to convert to numeric")
        # if isQuantatative(df):
            # raise TypeError(f"Series given is already quantatitive")
    # else:
    if (col is not None and isQuantatative(df[col])) or (col is None and isinstance(df, pd.Series) and isQuantatative(df)):
        raise TypeError("Series given is already quantatitive")

    if method == 'assign':
        log(f'Converting "{col}" to quantatative by assinging to arbitrary values', verbose)
        if isinstance(df, pd.Series):
            column, assings = pd.factorize(df)
            return (column, assings) if returnAssignments else column
        else:
            assert col is not None, 'Please provide column to assign'
            column, assings = pd.factorize(df[col])
            df[col] = column
            return (df, assings) if returnAssignments else df
    elif method == 'one_hot_encode':
        # This is all just overly-complicated parameter handling for 1 line of code
        skip = ensureIterable(skip)
        if col is not None:
            col = set(ensureIterable(col))
        else:
            if isinstance(df, pd.DataFrame):
                # col = set(df)
                col = set(catagorical(df).columns)
            else:
                return pd.get_dummies(df)

        for s in skip:
            col.remove(s)

        if isinstance(col, pd.Series):
            log(f'Converting "{df.name}" to quantatative by one hot encoding', verbose)
        else:
            log('Converting DataFrame to quantatative by one hot encoding', verbose)

        return pd.get_dummies(df, columns=list(col))
    else:
        raise TypeError(f"Bad method arguement '{method}' given to convert_numeric")

def split(*data, amt=.2, method:Union['random', 'chunk', 'head', 'tail']='random', target=[], splitTargets=False, seed=42):
    """ Splits the given data, both into train/test sets, and by taking out targets at the same time
        `target` can be a string or an iterable
        If `splitTargets` is set to False, the targets will always return DataFrames, even if
            they only have 1 column
        If you pass in multiple items for data, AND specify a target feature[s], then all the items
            must have the target columns
        The order goes:
            train_X, test_X, train_X1, test_X1, ..., train_y, test_y, train_y1, test_y1
            where it continues adding data and target splits in the order they are given.
            Simply put, it outputs in the same order you input the parameters as much as possible.
            Don't give multiple data AND split targets at the same time. While it can do it,
                it's simply too confusing to think through the order of the returned parameters.
        Setting the `method` to 'chunk' is the same as setting it to 'tail'.
    """
    if len(ensureIterable(data)) > 1 and len(target):
        warn("Please don't give multiple data AND split targets at the same time. While it can do it, "
             "it's simply too confusing to think through the order of the returned parameters.")
    # Pop the targets and combine everything into 1 ordered list of things we need to split
    splitMe = []
    for d in ensureIterable(data):
        d = d.copy()

        targets = [d.pop(t) for t in ensureIterable(target)]
        if splitTargets:
            splitMe += targets
        else:
            splitMe.append(pd.DataFrame(dict(zip(ensureIterable(target), targets))))
        # It makes more sense to do data, then target, not target then data
        splitMe.insert(0 if len(targets) else len(splitMe), d)

    # Now split everything in the list (order is important!)
    if method == 'random':
        return skms.train_test_split(*splitMe, test_size=amt, random_state=seed)
    elif method in ('head', 'tail', 'chunk'):
        rtn = []
        for d in splitMe:
            # Head an tail splitting are the same, just with opposite amts
            split = round(len(d) * (amt if method == 'head' else (1-amt)))
            rtn += [d.iloc[:split], d.iloc[split:]]
        return rtn
    else:
        raise TypeError("Invalid method parameter given")

# The main functions
def explore(data,
            target=None,
            stats=None,
            additionalStats=[],
            missing=True,
            corr=.55,
            entropy=None,
            start='Description',
            startFeature=None,
            startx=None,
            starty=None,
            startHue=None,
            alpha=None,
            ):
    # Parse params and make sure all the params are valid
    assert not isinstance(target, (list, tuple)), 'There can only be 1 target feature'
    assert target is None or target in data.columns, f'Target {target} is not one of the features'
    assert startFeature is None or startFeature in data.columns, 'startFeature must be a valid column name'
    assert len(data), 'DataFrame cannot be empty'

    if stats is None:
        stats = ['mean', 'median', 'std', 'min', 'max']
    if stats:
        stats += ensureIterable(additionalStats, True)
    if startFeature is None:
        if target is not None:
            startFeature = target
        else:
            startFeature = data.columns[0]

    # Define variables
    whatTheHeck = (corr, missing)
    max_name_len = len(max(data.columns, key=len))
    ALPHA = min(1, 1000/len(data)) if alpha is None else alpha

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
                    value=startFeature,
                    description='Feature',
                )
    featureBox.layout.visibility = 'hidden'
    featureABox = widgets.Dropdown(
        options=list(data.columns),
        value=startx if startx is not None else startFeature,
        description='x',
    )
    featureABox.layout.visibility = 'hidden'
    featureBBox = widgets.Dropdown(
        options=list(data.columns),
        value=starty if starty is not None else startFeature,
        description='y',
    )
    featureBBox.layout.visibility = 'hidden'
    featureHueBox = widgets.Dropdown(
        options=list(data.columns) + ['None'],
        value=startHue if startHue is not None else 'None',
        description='hue',
    )
    featureHueBox.layout.visibility = 'hidden'
    outlierSlider = widgets.FloatSlider(
        value=3,
        min=0.,
        max=10.,
        step=0.1,
        description='Z-Score:',
        # disabled=False,
        continuous_update=CONTINUOUS_UPDATE_SLIDER,
        # orientation='horizontal',
        # readout=True,
        # readout_format='.1f',
    )
    outlierSlider.layout.visibility = 'hidden'

    # All the actual logic
    def output(page, feature, a, b, hue, zscore):
        # See baffled comment above
        corr, missing = whatTheHeck
        featureBox.layout.visibility = 'hidden'
        featureABox.layout.visibility = 'hidden'
        featureBBox.layout.visibility = 'hidden'
        featureHueBox.layout.visibility = 'hidden'
        outlierSlider.layout.visibility = 'hidden'
        # Clear the output (because colab doesn't automatically or something?)
        clear_output(wait=True)

        plt.xticks(rotation=45)

        # match page:
        if   page == 'Description':
                print(f'There are {len(data):,} samples, with {len(data.columns)} columns:')
                print()
                print(', '.join(data.columns))
                print()
                print('which have types:')
                display(getNiceTypesTable(data))

                if len(quantitative(data)):
                    print('\nThe possible values for the Catagorical values:')
                    # This is just an overly complicated way to print them all nicely
                    for key, value in sort_dict_by_value_length(dict([(c, data[c].unique()) for c in catagorical(data).columns])).items():
                        # If it has too high of a cardinality, just print the first few
                        card = len(value)
                        shortened = False
                        if card > 30:
                            shortened = True
                            value = value[:29]

                        print(key + ":")
                        joined_list = ", ".join(value)
                        if len(joined_list) <= 80:  # adjust this number as needed
                            print('   ' + joined_list)
                        else:
                            for item in value:
                                print('   ' + item)
                        if shortened:
                            print(f'... ({card - 29} more catagories)')
        elif page == 'Stats':
                if len(quantitative(data)):
                    # print('Summary of Quantatative Values:')
                    display(data.agg(dict(zip(quantitative(data), [stats]*len(data.columns)))))
        elif page == 'Entropy':
                todo('Calculate entropy relative to the target feature')
                # if target is not None:
                # base = e if entropy is not None else entropy
                for c in data.columns:
                    print(f'The entropy of {c:>{max_name_len}} is: {round(scipy.stats.entropy(data[c].value_counts(normalize=True), base=entropy), 3)}')
                    # print(f'The entropy of {c} is: {entropy(data[c], data[target])}')
                # else:
                    # print('Target feature must be provided in order to calculate the entropy')
        elif page == 'Duplicates':
                todo()
        elif page == 'Head':
            with pd.option_context('display.max_columns', None):
                display(data.head())
        elif page == 'Counts':
                # This is sorted just so the features with less unique options go first
                for i in sorted(catagorical(data), key=lambda c: len(data[c].unique())):
                    print(f'{i} value counts:')

                    if len(data[i].unique()) == len(data[i]):
                        print('\tEvery sample has a unique catagory')
                    else:
                        print(pretty_counts(data[i]))
        elif page == 'Correlations':
                if len(quantitative(data)):
                    print('Correlations Between Quantatative Values:')
                    if type(corr) is bool:
                        display(quantitative(data).corr())
                    elif isinstance(corr, (int, float)):
                        corr = normalizePercentage(corr)
                        # Ignore if they're looking for a negative correlation, just get both
                        corr = abs(corr)
                        _corr = significantCorrelations(quantitative(data), corr)
                        if len(_corr):
                            a_len = max([len(i[0]) for i in _corr])
                            b_len = max([len(i[1]) for i in _corr])
                            for a,b,c in _corr:
                                print(f'\t{a:<{a_len}} <-> {b:<{b_len}}: {round(c, 2):+}')
                        else:
                            print(f'\tThere are no correlations greater than {corr:.0%}')
        elif page == 'Missing':
                # if len(_relevant):
                print('Missing Percentages:')
                if type(missing) is bool:
                    percent = data.isnull().sum()/len(data)*100
                    # This works, but instead I decided to overcomplicate it just so I can indent it
                    # print(percent)
                    print(pretty_2_column_array(percent))
                elif isinstance(missing, (int, float)):
                    missing = normalizePercentage(missing)
                    _missing = missingSummary(data, missing/100)
                    if len(_missing):
                        display(_missing)
                    else:
                        print(f'\tAll values are missing less than {missing:.0%} of their entries')
                else:
                    raise TypeError('Missing is a bad type')
        elif page == 'Features':
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
                    print(f'It has an entropy of {scipy.stats.entropy(data[feature].value_counts(normalize=True), base=entropy):.3f}', end=', ')
                    print(f'and a cardinaltiy of {len(data[feature].unique())}')
                    print('Value counts:')
                    print(pretty_counts(data[feature], paren=True))

                    sns.histplot(data[feature])

                # Quantative description
                else:
                    # Set the slider variables
                    outlierSlider.layout.visibility = 'visible'
                    # Todo This breaks on time data (I think)
                    # Todo This is usable, but can definitely be improved
                    if data[feature].std() > 1:
                        outlierSlider.max = abs(data[feature].max()) / data[feature].std()
                    else:
                        outlierSlider.max = abs(data[feature].max()) * data[feature].std()

                    correlations = []
                    for a, b, c in significantCorrelations(quantitative(data), corr):
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
                    # Because dates are weird
                    try:
                        print(f'It has a kurtosis value of {scipy.stats.kurtosis(data[feature]):,.2f}.')
                        print('\tNegative values mean less outliers than a normal distrobution, positive values mean more.')
                    except np.core._exceptions.UFuncTypeError: pass
                    print(f'It has a minimum value of {data[feature].min():,.2f}, and a maximum value of {data[feature].max():,.2f}.')
                    print(correlations)

                    # sns.scatterplot(data=data[feature])
                    # plt.show()
                    # display(interactWithOutliers(data, feature))

                    # def interactWithOutliers(df, feature=None, step=.2):
                    # widgets.interactive(
                    print()
                    showOutliers(data, feature, zscore, alpha=ALPHA)
                        # data=widgets.fixed(df),
                        # column=list(df.columns) if feature is None else widgets.fixed(feature),
                        # zscore=(0., df[feature].max() / df[feature].std(), step) if feature is not None else (0., 10, step)
                        # zscore=(0., 20, step)
                    # )

                print()
                # todo('Add nice plots here: scatterplots, histograms, and relating to the target feature')
        elif page == 'General Plots':
                if len(quantitative(data)):
                    print('Plot of Quantatative Values:')
                    sns.catplot(data=quantitative(data))
                    plt.show()
                if len(catagorical(data)):
                    print('Plot of Catagorical Value Counts:')
                    todo('catagorical (count?) plots')
                    # plt.show()
        elif page == 'Custom Plots':
                featureABox.layout.visibility = 'visible'
                featureBBox.layout.visibility = 'visible'
                featureHueBox.layout.visibility = 'visible'

                graph = sns.scatterplot(x=data[a], y=data[b], hue=None if hue == 'None' else data[hue], alpha=ALPHA)
                if isQuantatative(data[a]) and isQuantatative(data[b]):
                    try:
                        graph.set(title=f'Correlation: {data.corr()[a][b]:0.1%}')
                    except KeyError:
                        print('Cant calculate the correlations of dates for some reason')
                else:
                    # counts = data.groupby(a)[b].value_counts()
                    # print(counts.index.max())
                    # print(counts)
                    graph.set(title='Most common together: Todo')

                plt.show()
        elif page == 'Matrix':
                if len(quantitative(data)):
                    print('Something Something Matrix:')
                    if target in quantitative(data):
                        sns.pairplot(data=quantitative(data), hue=target)
                    else:
                        sns.pairplot(data=quantitative(data))
                    plt.show()
        elif page == 'Alerts':
                # TODO:
                # Check that entropy isn't too low
                # check that relative entropy isn't too low
                # check for spikes and plummets
                # high correlations between features
                # Print the kurtosis score with the outlier stuff

                # Check if our dataset is small
                if data[feature].count() < SMALL_DATASET:
                    print(f"Your dataset isn't very large ({data[feature].count()}<{SMALL_DATASET})")

                # Check the cardinality
                for c in catagorical(data):
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
                for q in quantitative(data):
                    try:
                        upper = data[q].max() - data[q].quantile(.75)
                        upperMid = data[q].quantile(.75) - data[q].median()
                        if upper - upperMid > OUTLIER_THRESHOLD * data[q].median():
                            print(f'Feature {q:>{max_name_len}} may have some upper outliers', end='   | ')
                            print(f'upper: {upper:>6.1f} | upperMid: {upperMid:>6.1f} | median: {data[q].median():>6.1f} | diff: {upper-upperMid:>6.1f}')

                        lower = data[q].quantile(.25) - data[q].min()
                        lowerMid = data[q].median() - data[q].quantile(.25)
                        if lower - lowerMid > OUTLIER_THRESHOLD * data[q].median():
                            print(f'Feature {q:>{max_name_len}} may have some lower outliers', end='   | ')
                            print(f'lower: {lower:>6.1f} | lowerMid: {lowerMid:>6.1f} | median: {data[q].median():>6.1f} | diff: {lower-lowerMid:>6.1f}')
                    except TypeError:
                        todo('checking dates for outliers isnt implemented')
        else:
                print('Invalid start option')

    # widgets.interact(output, page=combobox, feature=featureBox)
    ui = widgets.GridBox([combobox, featureABox, featureBox, featureBBox, outlierSlider, featureHueBox], layout=widgets.Layout(
                grid_template_columns='auto auto',
                grid_row_gap='10px',
                grid_column_gap='100px',
            )
       )
    out = widgets.interactive_output(output, {'page': combobox, 'feature': featureBox, 'a': featureABox, 'b': featureBBox, 'hue': featureHueBox, 'zscore': outlierSlider})
    display(ui, out)
quickSummary = explore

def suggestedCleaning(df, target):
    todo('suggestedCleaning')

def _cleanColumn(df, args, column, verbose, ignoreWarnings=False):
    global MODE_SELECTION
    missing = np.nan
    # We're allowing column to be None for the specific case of add_column (which doesn't require a column)
    if column in df.columns or column is None:
        for op, options in args.items():
            # Quick parameter type checking for bools
            if options is False and op not in ('missing_value', 'remove'):
                continue
            if options is True and not ignoreWarnings and op not in ('drop_duplicates', 'missing_value', 'remove', 'drop'):
                raise TypeError(f"'True' is an invalid option for {op} (for column {column})")

            if   op == 'drop_duplicates':
                warn('drop_duplicates hasnt been implemented yet for induvidual columns. What are you trying to do?')
            elif op == 'handle_outliers':
                zscore, method = options
                df[column] = handle_outliers(df[colulmn], method, zscore=zscore, verbose=verbose)
            elif op == 'replace':
                if not isinstance(options, dict):
                    raise TypeError(f"Please specify a dict for the replace option (under column {column})")
                log(f'Replacing specified entries in {column}', verbose)
                df[column] = df[column].replace(options)
            elif op == 'apply':
                if callable(options):
                    log(f'Applying function to {column}')
                    df[column] = df[column].apply(options, axis=1)
                else:
                    if not ignoreWarnings:
                        raise TypeError(f"Please specify a function to apply (under column {column})")
            elif op == 'missing_value':
                missing = options
            elif op == 'handle_missing':
                if options in ('mean', 'median') and isCatagorical(df[column]) and ignoreWarnings:
                    continue
                # This will throw the appropriate errors otherwise
                df[column] = handle_missing(df[column], method=options, missing_value=missing, verbose=verbose)
            elif op == 'queries':
                if options in ('mean', 'median') and isCatagorical(df[column]) and ignoreWarnings:
                    continue
                # If there's just one query, just accept it
                if len(options) == 2 and type(options[0]) is str:
                    options = [options]
                for q, method in options:
                    df = query(df, column, q, method, verbose=verbose)
            elif op == 'remove':
                df[column] = remove(df[column], options, verbose=verbose)
            elif op == 'bin':
                if isCatagorical(df[column]) and not ignoreWarnings:
                    warn(f'The bin option was set on "{column}", which is not quantatative, skipping.')
                else:
                    df[column] = bin(df[column], method, amt, verbose=verbose)
            elif op == 'normalize':
                if isCatagorical(df[column]) and not ignoreWarnings:
                    warn(f'The normalize option was set on {column}, which is not quantatative, skipping.')
                else:
                    df[column] = normalize(df[column], options, verbose=verbose)
            elif op == 'convert_numeric':
                if isQuantatative(df[column], time=False) and not ignoreWarnings:
                    warn(f'The conver_numeric option was set on {column}, which is not catagorical, skipping.')
                else:
                    df = convert_numeric(df, column, options, verbose=verbose)
            elif op == 'add_column':
                if isinstance(options, (tuple, list)):
                    if not isinstance(options[0], (tuple, list)):
                        options = [options]
                    for name, selection in options:
                        log(f'Adding new column "{name}"')
                        df[name] = selection
                else:
                    raise TypeError(f"add_column argument must be a tuple, or a list of tuples, not {type(options)}")
            elif op == 'drop':
                if options:
                    log(f'Dropping column "{column}"')
                    df = df.drop(columns=[column])
            else:
                raise TypeError(f'Invalid arguement {op} given')
    else:
        raise TypeError(f'Column "{column}" provided is not in the given DataFrame')

    return df

def clean(df:pd.DataFrame,
        config: Dict[str, Dict[str, Any]],
        verbose:bool=False,
        split:str=None,
        ) -> pd.DataFrame:
    """ Returns a cleaned copy of the DataFrame passed to it
        NOTE: The order of the entries in the config dict determine the order they are performed

        Arguments:
            config is a dict of this signature:
            NOTE: This is the suggested order
                {
                    # Do these to all the columns, or a specified column
                    'column/all': {
                        # Drop duplicate samples
                        ## Only applies to all
                        'drop_duplicates': bool,
                        # Removes samples which have a Z-score magnitude of greater than this value
                        'handle_outliers': Union[bool, Tuple[float, Union['remove', 'constrain']]],
                        # Maps feature values to a dictionary
                        'replace': Union[bool, Dict],
                        # Applies a function to the column
                        'apply': Union[bool, Callable],
                        # A list of (query, replacements).
                        ## If a Series is given, it will replace those values with the values at it's corresponding index
                        ## 'random' replaces values with either a random catagory, or a random number between min and max
                        ## 'balanced_random' replaces values with either a randomly sampled catagory (sampled from the column
                        ## itself, so it's properly biased), or a normally distributed sample
                        'queries': Union[bool, List[Tuple[str, Union[Series, 'remove', 'mean', 'median', 'mode', 'random', 'balanced_random', Any]]]],
                        # A ndarray of shape (1, n) of values to create a new column with the given name
                        ## Calling from a specific column has no effect, behaves the same under all
                        'add_column': Union[Tuple[str, np.ndarray], List[Tuple[str, np.ndarray]]],
                        # Specifies a value that is equivalent to the feature being missing
                        'missing_value': Any,
                        # Specifies a method by which to transform samples with missing features. Acts just like queries, but with missing values specifically
                        'handle_missing': Union[bool, Series, 'remove', 'mean', 'median', 'mode', 'random', 'balanced_random', Any],
                        # Removes all samples with the given value
                        'remove': Union[bool, Any],
                        # Specifies a method by which to bin the quantative value, or specify custom ranges
                        'bin': Union[bool, Tuple['frequency', int], Tuple['width', int], Iterable],
                        # Specifies a method by which to normalize the quantative values
                        'normalize': Union[bool, 'min-max', 'range'],
                        # Specifies a method by which to convert a catagorical feature to a quantative one
                        'convert_numeric': Union[bool, 'assign', 'one_hot_encode'],
                        # Drop the column
                        'drop': bool,
                    },
                }
    """
    raise DeprecationWarning('This function is no longer supported and is likely to break')
    df = df.copy()
    log = lambda s: print(s) if verbose else None

    # Make sure the all section is done last (so if we're doing one hot encoding it doesn't throw errors)
    config = OrderedDict(config)
    if 'all' in config.keys():
        config.move_to_end('all')

    for column, args in config.items():
        log(f'Working on "{column}"')
        if column.lower() == 'all':
            # We only want to add new columns once (not inside the for loop)
            if 'add_column' in args:
                # We only want to call that command manually
                df = _cleanColumn(df, {'add_column': args['add_column']}, None, verbose)
                del args['add_column']

            # Dropping duplicates means something different on the scale of a single column
            # than it does applied to the whole table
            if 'drop_duplicates' in args:
                log('\tDropping duplicate samples')
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
    if split is not None:
        if split in df.columns:
            return df.drop(columns=split), df[split]
        else:
            raise TypeError('Provided feature not in the resulting data (did you drop it in the cleaning process by accident?)')
    else:
        return df

def resample(X, y, method:Union['oversample', 'undersample', 'mixed']='oversample', seed=None):
    # match method:
    if method == 'oversample':
        sampler = RandomOverSampler(random_state=seed)
        return sampler.fit_resample(X, y)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=seed)
        return sampler.fit_resample(X, y)
    elif method == 'mixed':
        todo('figure out how to mix under and over sampling')
    else:
        raise TypeError("Invalid method arguement given")

# @_cleaning_func(test=pd.Series)
@_cleaning_func(testPredictions=pd.Series)
def evaluateQuantitative(test, testPredictions, train=None, trainPredictions=None, accuracy=3, explain=False, compact=False, line=False, log=...):
    """ Evaluate your predictions of an ML model.
        NOTE: compact overrides explain.
     """
    assert (train is None) == (trainPredictions is None), 'You have to pass both train & trainPredictions'
    # display(test)
    # display(testPredictions)
    # @_cleaning_func() SHOULD handle this
    test = pd.Series(test)
    testPredictions = pd.Series(testPredictions)

    def _score(name, func, explaination, _test=True, **kwargs):
        name += ':'
        if compact:
            print(f'{name} {func(test, testPredictions, **kwargs) if _test else func(train, trainPredictions, **kwargs):,.{accuracy}f}', end='   ')
        else:
            # print('~'*20, func(test, testPredictions, **kwargs) if _test else func(train, trainPredictions, **kwargs), '~'*20)
            print(f'\t{name:<23} {ensureNotIterable(func(test, testPredictions, **kwargs) if _test else func(train, trainPredictions, **kwargs)):,.{accuracy}f}')
            if explain:
                print('\t\t' + explaination)

    def _quantatative(_test=True):
        _score('Root Mean Square Error', mean_squared_error,  'An average of how far off we are from the target, in the same units as the target. Smaller is better.', _test, squared=False)
        _score('My own measure',         lambda a, b, **k: mean_squared_error(a, b, **k) / a.mean(),  'Root mean square / average value. Eliminates the domain a bit. Smaller is better.', _test, squared=False)
        _score('Mean Absolute Error',    mean_absolute_error, 'Similar to Root Mean Square Error, but better at weeding out outliers. Smaller is better.',             _test)
        _score('Median Absolute Error',  median_absolute_error, '',             _test)
        _score('R^2 Score',              r2_score,            'An average of how far off we are from just using the mean as a prediction. Larger is better.',          _test)

        def amtInPercent(truth, pred, precent):
            return ((truth - pred).abs() / truth <= (percent / 100)).values.sum() / len(truth) * 100

        for percent in (5, 10, 20, 50):
            _score(f'Within {percent}%', lambda a, b, **k: amtInPercent(a, b, percent), f'How many of the samples are within {percent}% of their actual values', _test)

    print('Test:')
    _quantatative()
    if train is not None and trainPredictions is not None:
        print('\nTrain:')
        _quantatative(False)

    if line:
        sns.set(rc={'figure.figsize':(11.7,8.27)})

        delta = test - testPredictions
        # display(testPredictions)
        # display(test)
        # display(delta)
        testfinal = pd.DataFrame({
            'Predictions': testPredictions,
            'Ground Truth': test,
            'difference': delta,
            'percent_difference': abs(delta/test),
            # 'percent_bucket': (test - testPredictions).abs() / test <= (percent / 100)#[ "above 20%" if i >= 0.2 else "below 20%" for i in testfinal.percent_difference ],
        })
        testfinal['percent_difference'] = bin(testfinal['percent_difference'], method=(0, .05, .10, .20, .50, 1))
        # display(testfinal['percent_difference'].iloc[0] == pd.Interval(0.5, 1.0, closed='right'))
        # display(testfinal['percent_difference'])

        testfinal['percent_difference'] = testfinal['percent_difference'].replace({
            pd.Interval(0, .05, closed='right'): 'Within 5%',
            pd.Interval(.05, .1, closed='right'): 'Within 10%',
            pd.Interval(.1, .2, closed='right'): 'Within 20%',
            pd.Interval(.2, .5, closed='right'): 'Within 50%',
            pd.Interval(0.5, 1.0, closed='right'): 'Within 100%',
        })
        # display(testfinal['percent_difference'])
        color_dict = dict({
            'Within 5%': 'tab:green',
            'Within 10%': 'tab:green',
            'Within 20%': 'tab:blue',
            'Within 50%': 'tab:orange',
            'Within 100%': 'tab:red',
            np.NaN: 'tab:red'
        })
        # Interval(0.5, 1.0, closed='right'), Interval(0.05, 0.1, closed='right'), Interval(0.2, 0.5, closed='right'), Interval(0.0, 0.05, closed='right'), Interval(0.1, 0.2, closed='right')

        # print(testfinal['abspercentmiss'].describe(percentiles=[.1,.2,.3,.4,.5,.6,.7,.8,.9,.95]))
        xlims = (0,1e3)
        # ylims=(0,1e3)
        ax = sns.scatterplot(data=testfinal,x='Ground Truth',y='Predictions',hue="percent_difference",palette=color_dict)
        # ax.set(xscale="log", yscale="log", xlim=xlims, ylim=ylims)
        ax.plot(xlims,xlims, color='r')
        # ax.plot(color='r')
        # plt.legend(labels=['perfect',"below 5",'above 5','10-20%','above 20'])
        plt.show()
evaluateQ = evaluateQuantitative

def evaluateCatagorical(test, testPredictions, train=None, trainPredictions=None, accuracy=3, curve=False, confusion=False, explain=False, compact=False):
    """ Evaluate your predictions of an ML model.
        NOTE: compact overrides explain.
     """
    assert (train is None) == (trainPredictions is None), 'You have to pass both train & trainPredictions'

    def _score(name, func, explaination, _test=True, **kwargs):
        name += ':'
        if compact:
            print(f'{name} {func(test, testPredictions, **kwargs) if _test else func(train, trainPredictions, **kwargs):,.{accuracy}f}', end='   ')
        else:
            print(f'\t{name:<23} {func(test, testPredictions, **kwargs) if _test else func(train, trainPredictions, **kwargs):,.{accuracy}f}')
            if explain:
                print('\t\t' + explaination)

    def _catagorical(_test=True):
        # Can't do an F1 score with more than 2 classes
        try:
            _score('F1',        sklearn.metrics.f1_score,        'F1 is essentially an averaged score combining precision and recall',            _test)
        except ValueError:
            pass
        _score('Accuracy',  sklearn.metrics.accuracy_score,  'Accuracy is a measure of how well the model did on average',                    _test)
        _score('Precision', sklearn.metrics.precision_score, 'Precision is a measure of how many things we said were true and we were wrong', _test)
        _score('Recall',    sklearn.metrics.recall_score,    'Recall is a measure of how many things we missed out on',                       _test)

    print('Test:')
    _catagorical()

    if confusion:
        ConfusionMatrixDisplay.from_predictions(test, testPredictions, cmap='Blues')
        plt.show()
    if curve:
        PrecisionRecallDisplay.from_predictions(test, testPredictions)
        plt.show()

    if train is not None and trainPredictions is not None:
        print('\nTrain:')
        _catagorical(False)

        if confusion:
            ConfusionMatrixDisplay.from_predictions(train, trainPredictions, cmap='Blues')
            plt.show()
        if curve:
            PrecisionRecallDisplay.from_predictions(train, trainPredictions)
            plt.show()
evaluateC = evaluateCatagorical

def evaluate(catagorical, test, testPredictions, train=None, trainPredictions=None, accuracy=3, curve=False, confusion=False, explain=False, compact=False, line=False):
    """ Evaluate your predictions of an ML model.
        NOTE: compact overrides explain.
     """
    assert (train is None) == (trainPredictions is None), 'You have to pass both train & trainPredictions'
    raise DeprecationWarning('Please use evaluateQ or evaluateC instead')

    def _score(name, func, explaination, _test=True, **kwargs):
        name += ':'
        if compact:
            print(f'{name} {func(test, testPredictions, **kwargs) if _test else func(train, trainPredictions, **kwargs):,.{accuracy}f}', end='   ')
        else:
            print(f'\t{name:<23} {func(test, testPredictions, **kwargs) if _test else func(train, trainPredictions, **kwargs):,.{accuracy}f}')
            if explain:
                print('\t\t' + explaination)

    # def _catagorical(_test=True):
    #         print(f'\t{name:<23} {func(test, testPredictions, **kwargs) if _test else func(train, trainPredictions, **kwargs):,.{accuracy}f}')
    #         if explain:
    #             print('\t\t' + explaination)

    def _catagorical(_test=True):
        _score('F1',        sklearn.metrics.f1_score,        'F1 is essentially an averaged score combining precision and recall',            _test)
        _score('Accuracy',  sklearn.metrics.accuracy_score,  'Accuracy is a measure of how well the model did on average',                    _test)
        _score('Precision', sklearn.metrics.precision_score, 'Precision is a measure of how many things we said were true and we were wrong', _test)
        _score('Recall',    sklearn.metrics.recall_score,    'Recall is a measure of how many things we missed out on',                       _test)

    def _quantatative(_test=True):
        _score('Root Mean Square Error', mean_squared_error,  'An average of how far off we are from the target, in the same units as the target. Smaller is better.', _test, squared=False)
        _score('My own measure',         lambda a, b, **k: mean_squared_error(a, b, **k) / a.mean(),  'Root mean square / average value. Eliminates the domain a bit. Smaller is better.', _test, squared=False)
        _score('Mean Absolute Error',    mean_absolute_error, 'Similar to Root Mean Square Error, but better at weeding out outliers. Smaller is better.',             _test)
        _score('Median Absolute Error',  median_absolute_error, '',             _test)
        _score('R^2 Score',              r2_score,            'An average of how far off we are from just using the mean as a prediction. Larger is better.',          _test)
        for percent in (5, 10, 20, 50):
            _score(f'Within {percent}%', lambda a, b, **k: amtInPercent(a, b, percent), f'How many of the samples are within {percent}% of their actual values', _test)

        def amtInPercent(truth, pred, precent):
            combined = pd.concat([truth, pred], axis=1)
            combined.columns = ["truth", "pred"]
            combined["absdiff"] = (combined["truth"] - combined["pred"]).abs()
            combined["absdiff_pct"] = combined["absdiff"] / combined["truth"]
            return len(combined[combined["absdiff_pct"] <= (percent / 100)]) / len(combined) * 100

    # Catagorical measures
    if catagorical:
        print('Test:')
        _catagorical()

        if confusion:
            ConfusionMatrixDisplay.from_predictions(test, testPredictions, cmap='Blues')
            plt.show()
        if curve:
            PrecisionRecallDisplay.from_predictions(test, testPredictions)
            plt.show()

        if train is not None and trainPredictions is not None:
            print('\nTrain:')
            _catagorical(False)

        if confusion:
            ConfusionMatrixDisplay.from_predictions(train, trainPredictions, cmap='Blues')
            plt.show()
        if curve:
            PrecisionRecallDisplay.from_predictions(train, trainPredictions)
            plt.show()
    # Quantative measures
    else:
        print('Test:')
        _quantatative()
        if train is not None and trainPredictions is not None:
            print('\nTrain:')
            _quantatative(False)

        if line:
            sns.set(rc={'figure.figsize':(11.7,8.27)})
            color_dict = dict({'below 20%':'tab:blue',
                                'above 20%': 'tab:orange'})

            shower = pd.DataFrame(student_ds, columns=['predictions'])
            shower.columns = ['predictions']
            testfinal = pd.concat([shower,targets['actual']],axis=1)
            testfinal['difference'] = testfinal['actual']-testfinal['predictions']
            testfinal['percent_difference'] = abs(testfinal['difference']/testfinal['actual'])
            testfinal['percent_bucket'] = ["above 20%" if i >= 0.2 else "below 20%" for i in testfinal.percent_difference]

            # print(testfinal['abspercentmiss'].describe(percentiles=[.1,.2,.3,.4,.5,.6,.7,.8,.9,.95]))
            # xlims=(0,1e3)
            # # ylims=(0,1e3)
            ax = sns.scatterplot(data=testfinal,x='actual',y='predictions',hue="percent_bucket",palette=color_dict)
            # ax.set(xscale="log", yscale="log", xlim=xlims, ylim=ylims)
            # ax.plot(xlims,xlims, color='r')
            ax.plot(color='r')
            # plt.legend(labels=['perfect',"below 5",'above 5','10-20%','above 20'])
            plt.show()
            print("-"*77)
            print("\n"*3)


"""
TODO: Add cross-validation to evaluate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# X = ... # training data
# y = ... # target variable

# model = LogisticRegression()
# scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores) * 2))

 """


def importances(tree, names=None, rtn=False, graph=True, best=.01):
    if names is None:
        names = tree.feature_names_in_
    df = pd.DataFrame({
        'feature': names,
        'importance': tree.feature_importances_
    })

    if best:
        # df = df.assign(best=df.importance > best)
        df = df.loc[df.importance >= best]

    df = df.sort_values(by='importance', ascending=False, axis=0)
    if graph:
        sns.catplot(data=df, x='importance', y='feature', kind='bar', height=10, aspect=2)
        plt.show()

    if rtn:
        return df

def saveStats(file, name, model, testY, predY, trainY=None, trainPredY=None, notes='', new=False, show=True, save=True):
    def doit():
        print(name + ':')
        print(notes)
        print()
        print('Model type:', type(model))
        print('Parameters:')
        for key, val in model.get_params().items():
            print(f'\t{key}: {val}')
        print('\nImportances:')
        print(importances(model, rtn=True, graph=False))
        print('\nStats:')
        evaluate(testY, predY, trainY, trainPredY, compact=False)
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')

    with open(file, 'w' if new else 'a') as f:
        with redirect_stdout(f):
            doit()
    if show:
        doit()

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(history['index'], history['loss'], label='Train Loss')
    plt.plot(history['index'], history['val_loss'], label='Value Loss')
    plt.legend()
    plt.show()
