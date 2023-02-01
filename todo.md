# Quick Summary
- Add a start parameter
- add a bunch of value_count(normalize=True) calls
- make the unique section be formatted better
- add relplot with hue parameter
- add a generalized catagorical graph
- add hue to matrix graph
- if there are no catagorical values, remove the unique print label
- Fix all the commended out parts
- Convert all the print statements to be graph titles instead
- In summary, add info on the amounts of unique values of the target (for resampling)
- Look for extremes in the data
    - if there are (or heck, if not) look for big spikes or pluments in the data (may require graphs)
- in base summary tab, get the initial target range (ie without doing anything, we can get x% yes)
- make jupyter tabs/accordian work??
- Make the background not white
- add a loading bar for matrix plot and other plots
- add induvidual plots with a combobox selecting which ones
- add an entropy threshold option
- detect unusual clumps of values

# Clean
- make a query option in clean function
- add an option to handle unknowns by randomly insterting random values
    - And another option to insert random values biased by how they are distributed in the data
- fix it


# Ensemble
- A function which takes a function which returns a model, and an amount, and creates a bunch of models and aggrigates the answers
