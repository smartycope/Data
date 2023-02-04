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
    - mode[s], std, quantative entropy, catagorical correlations, data.groupby(feature)[target].value_counts()
- Alerts for:
    - Check that entropy isn't too low
    - check that relative entropy isn't too low
    - check for spikes and plummets
    - high correlations between features

# SuggestedCleaning
- A function that does a bunch of math and returns a suggested clean config

# Clean
- Make add column a list so we can do it more than once

# Resample
- It should work, but I haven't tested it yet
