# Predicting drug predilections from personality

The University of California at Irvine has a [repository](http://archive.ics.uci.edu/ml/datasets.php) of various datasets for machine learning applications, [one of which](http://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29#) is concerned with the consumption of drugs from 1885 respondents. 

Data about 12 attributes are gathered from each respondent, ranging from personality type to country of residence and ethnicity. Then questions are asked about eighteen legal and illegal drugs and when the respondent last used them. A detailed study by the original authors of the dataset can be found [here](https://www.researchgate.net/publication/338737362_Personality_Traits_and_Drug_Consumption_A_Story_Told_by_Data?).


## Proposal

Can we answer which drug(s) a participant is most likely to use, based solely on their personality?

Or can we infer personality based on drug use?

## Methods

It was decided that the data be distilled into a binary classification problem - namely the probability that a respondent used a certain drug in the last month or not. 

A four-level network was created using XGBoost classifiers, one classifier for each drug. The classifiers in one level feed into the next in the hopes that additional closely correlated features could assist the classifier in its decision. 

## Results

A Heroku-hosted Flask app was created in order to visualize the results of the modelling process. It can be found [here](https://dry-stream-60533.herokuapp.com/).