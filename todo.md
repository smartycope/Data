# Quick Summary
- add feature specific graphs under the features menu
- add hue to matrix graph
- bin quantative features and graph them as a histogram?
- Add titles to graphs
- Generally just make graphs prettier
- Look for extremes in the data
    - if there are (or heck, if not) look for big spikes or pluments in the data (may require graphs)
- detect unusual clumps of values
- in base summary tab, get the initial target range (ie without doing anything, we can get x% yes)
- maybe make a target-target parameter? (the value we're looking for target to be?)
- make jupyter tabs/accordian work??
- Make the background not white -- general styling
- add a loading bar for matrix plot and other plots
- add an entropy threshold option
- add these to the feature specific menu:
    - model\[s], std, quantative entropy, catagorical correlations, data.groupby(feature)\[target].value_counts()
- Alerts for:
    - Check that entropy isn't too low
    - check that relative entropy isn't too low
    - check for spikes and plummets
    - high correlations between features
    - kurtosis value being too high/low
    - Not very many items in a catagory
    - Check for duplicate samples
- add boxplots
- add counts of quantative features

# SuggestedCleaning
- A function that does a bunch of math and returns a suggested clean config

# Clean
- converting date/time type to "days since \[param]"
    - option to just use the smallest date
    - option to use POSIX standard
- consider changing it to do all in order instead of always last
- better errors (give where the errors are)
- add new sample option somehow?
    - like so you don't have to make a new config for a holdout set
- Split this up into a ton of mini functions instead
- bin by month function
- add inplace parameters to all the functions (and maybe have them defualt to True?)

# Resample
- Add mixed option (and a tradeoff parameter)

# Evaluate
- Quantative
    - abs(mean(error))
    - abs(median(error))
    - % within x% for x in (5,10,20,50)
    - RMSE
    - Use this for reference:
        - https://colab.research.google.com/github/byui-cse/cse450-course/blob/master/notebooks/module03_housing_grading_mini.ipynb


- useful function: from calendar import month_abbr
    - df.at()

- dont be afraid to log or square feetures
    - we like linear data better than exponential data
    - try to make data linear
- Throw *EVERYTHING* you can possibly think of adn add it to the model
- make more graphs just piddling around and playing with features
    - make silly graphs and find crazy correlations



- sns.heatmap of correlations is WAY better

RandomizedSearchCV()
    - give ranges
    - params = RandomizedSearchCV.best_estimator_.get_params()

- order of the columns matter in xgboost
    - shuffle order of features when randomizing


- Explain extreme outliers
