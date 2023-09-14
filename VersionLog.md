# Popular-Stock-Sentiment-Analysis VERSION LOG

1.0.0: Early Release Version 

- Works.

1.0.1:

- Added a function that checks the overall ensemble's accuracy.

- Changed evaluation for accuracy to evaluate.load due to load_metric's deprecation.

- Fixed a bug when no checkboxes are checked, none of the models got sent to the classifier, hence no classification happened.

- Fixed a bug where an empty string was sent instead of the models due to no default option for the dropdown menu, hence no classification happened.

- Fixed data leakage that ocurred due to mixing of test and training set data, modularized test set formatting and creation.	

- Fixed a bug where the ensemble was not properly setting the weights of the models.

- Changed the train-test split to be a 75% - 25% split instead of a 80% - 20% split. Also increased the subset size of the movie_reviews dataset to be 60% of the dataset included instead of only 50%.

- Fixed a bug where bigrams where only only being applied partially for the dataset in the Naive Bayes Classifier.

1.1:

- Completely refactored the Sentiment Analyzer class, now it is composed of a parent Abstract Base Class and subclasses that correspond to each type of model that can be initialized in the Sentiment Analyzer class, which manages what actually gets initialized. This greatly improves Maintainability(You just need to make a new subclass to add a new model and changes are less likely to break the whole thing) and Readability(model functions are much shorter because they all pertain to one subclass and also easier to find since they are not all in one big class).

- Added the ability to save models for all models, to save on training time. Also ensured the train_test split is the same across runs to avoid leakage when saving models. 

- Added Two new model options into the Sentiment Analyzer pipeline that can be chosen in the front-end.

- Added new subreddit options for initial webscrapping.

- Modified mention table in database to avoid duplicate text when scraping financial news and added a source column to the tables, specifying whether the mention was a financial news article or a subreddit text.

- Now scrape new instead of top on subreddits. This is because some subreddits do not have many posts on top, so new pings more responses and is also more recent.
