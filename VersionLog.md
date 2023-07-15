# Popular-Stock-Sentiment-Analysis VERSION LOG

1.0.0: Early Release Version 

- Works

1.0.1:

- Added a function that checks the overall ensemble's accuracy
- Changed evaluation for accuracy to evaluate.load due to load_metric's deprecation.
- Fixed a bug when no checkboxes are checked, none of the models got sent to the classifier, hence no classification happened.
- Fixed a bug where an empty string was sent instead of the models due to no default option for the dropdown menu, hence no classification happened.
- Fixed data leakage that ocurred due to mixing of test and training set data.	
- Fixed a bug where the ensemble was not properly setting the weights of the models.
- Changed the train-test split to be a 75% - 25% split instead of a 80% - 20% split. Also increased the subset size of the movie_reviews dataset to be 60% of the dataset included instead of only 50%.
- Fixed a bug where bigrams where only only being applied partially for the dataset in the Naive Bayes Classifier