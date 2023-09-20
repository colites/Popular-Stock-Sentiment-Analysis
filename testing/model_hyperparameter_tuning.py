import os
import pandas as pd
import psycopg2
import nltk
from nltk.corpus import movie_reviews, stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import database_config

####### WARNING THIS COULD TAKE DAYS TO RUN, EVEN ON GOOD PC's ###############################
####### THIS IS MEANT AS AN EXHAUSTIVE PARAMETERS-TUNING TEST FOR THE SCIKIT-LEARN MODELS ##############################

def save_mentions_csv():
    """
    Export mention data from the database to a CSV file.
    """

    try:
        conn = psycopg2.connect(host = database_config.DB_HOST,
                                       database = "hype_stock",
                                       user = database_config.DB_USER,
                                       password = database_config.DB_PASS)

        query = """
            SELECT mention_text, source_type
            FROM mentions
        """

        # Read the query and export to a csv
        df = pd.read_sql(query, conn)
        df.to_csv("custom_dataset.csv", index=False)
        
    except psycopg2.Error as e:
        print(f"Error querying the database: {e}")

    if conn:
        conn.close()


def format_custom_dataset_scikit():
    """
    Format and preprocess the custom dataset for scikit-learn models.

    Returns:
        Tuple: A tuple containing train and test data, each with features and labels.
    """

    stop_words = set(stopwords.words('english'))
    df = pd.read_csv("custom_dataset.csv")

    # Shuffles and splits the DataFrame directly
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

    # Extract features and labels from the split DataFrames
    train_features = train_df["mention_text"].tolist()
    train_labels = train_df["label"].tolist()
    test_features = test_df["mention_text"].tolist()
    test_labels = test_df["label"].tolist()

    # Preprocessing for the features
    filtered_train_features = []
    for text in train_features:
        tokenized_text = nltk.word_tokenize(text)
        filtered_text = ' '.join([word for word in tokenized_text if word.lower() not in stop_words])
        filtered_train_features.append(filtered_text)

    filtered_test_features = []
    for text in test_features:
        tokenized_text = nltk.word_tokenize(text)
        filtered_text = ' '.join([word for word in tokenized_text if word.lower() not in stop_words])
        filtered_test_features.append(filtered_text)
        
    return (filtered_train_features, train_labels), (filtered_test_features, test_labels)


def test_SVM():
    """
    Test Support Vector Machine (SVM) models with different configurations.

    Returns:
        dict: A dictionary containing the best hyperparameters and scores for LinearSVC and SVC models.
    """
     
    (train_features, train_labels), _ = format_custom_dataset_scikit()

    # LinearSVC testing preparation
    linear_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', LinearSVC())
    ])

    # LinearSVC with TFID vectorizer pipeline
    linear_pipeline_vec = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', LinearSVC())
    ])
    
    linear_param_grid = {
                        'classifier__C': [0.01, 0.1, 1, 10, 100, 1000],
                        'classifier__penalty': ['l1', 'l2'],
                        'classifier__loss': ['hinge', 'squared_hinge'],
                        'classifier__dual': [True, False],
                        'classifier__class_weight': [None, 'balanced'],
                        'classifier__max_iter': [1000, 3000]}

    # SVC testing preparation
    svc_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', SVC())
    ])
    
    svc_param_grid = {
                    'classifier__C': [0.1, 1, 10, 100],
                    'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
                    'classifier__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                    'classifier__degree': [2, 3],  
                    'classifier__coef0': [0, 1, 2],
                    'classifier__shrinking': [True, False],
                    'classifier__class_weight': [None, 'balanced']}
    
    # GridSearchCV for LinearSVC
    grid_search_linear = GridSearchCV(linear_pipeline, linear_param_grid, cv=5)
    grid_search_linear.fit(train_features, train_labels)
    print("\n--- LinearSVC Results ---")
    print("Best hyperparameters: ", grid_search_linear.best_params_)
    print("Best score: ", grid_search_linear.best_score_)

    # GridSearchCV for LinearSVC tfidvectorizer
    grid_search_linear = GridSearchCV(linear_pipeline_vec, linear_param_grid, cv=5)
    grid_search_linear.fit(train_features, train_labels)
    print("\n--- LinearSVC vec Results ---")
    print("Best hyperparameters: ", grid_search_linear.best_params_)
    print("Best score: ", grid_search_linear.best_score_)
    
    # GridSearchCV for SVC
    grid_search_svc = GridSearchCV(svc_pipeline, svc_param_grid, cv=5)
    grid_search_svc.fit(train_features, train_labels)
    print("\n--- SVC Results ---")
    print("Best hyperparameters: ", grid_search_svc.best_params_)
    print("Best score: ", grid_search_svc.best_score_)
    #Best hyperparameters:  {'classifier__C': 1, 'classifier__class_weight': None, 'classifier__coef0': 2, 'classifier__degree': 3, 'classifier__gamma': 0.01, 'classifier__kernel': 'poly', 'classifier__shrinking': True}
    # Gonna make this return just in case I ever decide to make automated hyper_parameter tuning in my model classes
    return {"LinearSVC": grid_search_linear.best_params_, "SVC": grid_search_svc.best_params_}

def test_RandomForest():
    """
    Test Random Forest classifier with different configurations.

    Returns:
        dict: A dictionary containing the best hyperparameters and score for the RandomForestClassifier.
    """

    (train_features, train_labels), _ = format_custom_dataset_scikit()

    # RandomForest testing preparation
    rf_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', RandomForestClassifier())
    ])
    
    rf_param_grid = {
                    'classifier__n_estimators': [10, 50, 100, 200, 350],
                    'classifier__max_depth': [None, 10, 20, 30, 40, 50],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__max_features': ['sqrt', 'log2'],
                    'classifier__bootstrap': [True, False],
                    'classifier__criterion': ['gini', 'entropy'],
                    'classifier__class_weight': [None, 'balanced', 'balanced_subsample']}
    
    # GridSearchCV for RandomForestClassifier
    grid_search_rf = GridSearchCV(rf_pipeline, rf_param_grid, cv=5)
    grid_search_rf.fit(train_features, train_labels)
    print("\n--- RandomForest Results ---")
    print("Best hyperparameters: ", grid_search_rf.best_params_)
    print("Best score: ", grid_search_rf.best_score_)

    return {"RandomForest": grid_search_rf.best_params_}

def test_LogisticRegression():
    """
    Test Logistic Regression classifier with different configurations.

    Returns:
        dict: A dictionary containing the best hyperparameters and score for the LogisticRegression.
    """
    
    (train_features, train_labels), _ = format_custom_dataset_scikit()

     # LogisticRegression testing preparation
    lr_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', LogisticRegression())
    ])

    lr_param_grid = {
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    'classifier__fit_intercept': [True, False],
                    'classifier__class_weight': [None, 'balanced'],
                    'classifier__max_iter': [100, 300, 1000]}
    
    # GridSearchCV for LogisticRegression
    grid_search_lr = GridSearchCV(lr_pipeline, lr_param_grid, cv=5)
    grid_search_lr.fit(train_features, train_labels)
    print("\n--- LogisticRegression Results ---")
    print("Best hyperparameters: ", grid_search_lr.best_params_)
    print("Best score: ", grid_search_lr.best_score_)

    return {"LogisticRegression": grid_search_lr.best_params_}  

if not os.path.exists("custom_dataset.csv"):
    save_mentions_csv()
else:
    SVM_hyperparameters = test_SVM()
    random_forest_hyperparameters = test_RandomForest()
    logistic_regression_hyperparameters = test_LogisticRegression()

