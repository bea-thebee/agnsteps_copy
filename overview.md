## Evaluating Error in Bayesian Block Testing Through Interval Variations
# The Issue
It is as of yet unknown how much error (especially related to false positives) is present in the analysis. It's incredibly hard to determine due to minimal other data sources to compare to and it being a new work in general. 
# Potential Solution
One potential solution is running comparison tests on the same source multiple times. Specifically observing the interval length. The initial test would be run on a given object with a given interval, say, 30 days. Then a large number of trials will be done with a number of intervals averaging on 30 days. This information will then be added to a dataframe and the measures of center will be compared from there. 
