"""
Helper functions for drug consumption modelling project
"""

def create_non_light_heavy_cats(drug_value, cohorts):
    """
    Cohorts is a list of ints, where the first item is the min value of first cohort, 
    and the second item is the min value of second cohort, etc.
    Thus for cohorts = [0,4,6]

    cohort 0 = all values 0-3
    cohort 1 = all values 4-5
    cohort 2 = all values 6+

    Takes in drug_value, which is an integer from 0-6 with the following classes:
    
    Possible values for drug_value:
    0: Never used before
    1: Used over a decade ago
    2: Used in last decade
    3: Used in last year
    4: Used in last month
    5: Used in last week
    6: Used in last day

    """

    for i in range(0,len(cohorts)):
        if i == len(cohorts) - 1:
            if drug_value >= cohorts[i]:
                return i
        if (drug_value >= cohorts[i]) and (drug_value < cohorts[i+1]):
            return i

