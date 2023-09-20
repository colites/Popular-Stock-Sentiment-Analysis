import functools
import os
import numpy as np
from abc import ABC, abstractmethod
import evaluate
import pandas as pd
import torch
import nltk
from joblib import dump, load
import nltk.classify.util, nltk.metrics
from nltk.corpus import movie_reviews, stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from transformers import TrainingArguments, Trainer, DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

"""
Sentiment Analysis Ensemble Module

This module contains classes and utilities for performing sentiment analysis on stock mentions. 
At the core is an ensemble sentiment analyzer which aggregates results from various models 
like SVM, NaiveBayes, Distilbert, Logistic Regression, and Random Forest. Each of these models 
can be activated or deactivated from the ensemble via user configuration. The ensemble approach 
is designed to provide a more robust sentiment prediction by leveraging the strengths of individual models.
"""

class SentimentAnalyzer:
    """
    A class to perform sentiment analysis using an ensemble of models.
    
    Attributes:
        model_classes (dict): A mapping of model names to their corresponding classes.
        train_set (list): List of training data.
        test_set (list): List of testing data.
        stop_words (set): Set of stopwords to be removed during preprocessing.
        models (dict): Dictionary of trained models.
        weights (dict): Dictionary of model weights.
    """
    
    def __init__(self, model_names=['SVM']):
        """
        Initializes the SentimentAnalyzer with the specified models.
        
        Loads or trains the specified models, prepares datasets, sets stopwords, 
        and initializes model weights.
        
        Parameters:
        - model_names (list of str): List of model names to be used. Defaults to ['SVM'].
        """
        
        self.model_classes = {
            'bayes': NaiveBayesModel,
            'SVM': SVMModel,
            'distilbert': DistilbertModel,
            'LR': LogisticRegressionModel,
            'RF': RandomForestModel
        }

        self.train_set, self.test_set = self.create_split()
        self.stop_words = set(stopwords.words('english'))

        # Make the directory where the models are saved
        if not os.path.isdir('./models'):
            os.makedirs('./models')
    
        self.models =  {model_name: self.model_classes[model_name](self.train_set, self.test_set, self.stop_words) for model_name in model_names}
        self.weights = {model_name : model.weight for model_name, model in self.models.items()}


    def create_split(self):
        """
        Reads the custom dataset and makes a train_test split of 80-20

        Returns:
            tuple: The train_set and test_set with word labels.
        """

        df = pd.read_csv("custom_dataset.csv")

        # Extract features and labels from the split DataFrames and make them into lists
        features_custom = df["mention_text"].tolist()
        labels_custom = df["label"].tolist()

        # Make the text and labels from the custom dataset into a list of tuples. 
        dataset_custom = []
        for i in range(len(features_custom)):
            dataset_custom.append((features_custom[i],labels_custom[i]))
                                  
        train_set_custom, test_set_custom = train_test_split(dataset_custom, test_size=0.20, shuffle=True, random_state=42)

        return train_set_custom, test_set_custom

            
    def predict_all(self, text):
        """
        Predict the sentiment for a given text using all loaded models and aggregate the results
        based on their weights.
        
        Args:
            text (str): The input text to be analyzed.
        
        Returns:
            str: The aggregated sentiment prediction.
        """
        
        sentiment_map = {'pos': 'positive', 'neg': 'negative', 'neu': 'neutral'}
        sentiment_weight = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for model_name, model in self.models.items():
            sentiment = model.predict(text)
            sentiment = sentiment_map[sentiment]
            sentiment_weight[sentiment] += self.weights[model_name]

        # Return the sentiment with the highest total weight
        return max(sentiment_weight, key=sentiment_weight.get)

    
    def evaluate_ensemble(self):
        """
        Evaluates the ensemble classifier's performance based on the models present in the ensemble. 
        The method uses the models' weights to predict sentiment and calculates the overall accuracy of the ensemble.

        Returns:
            float: The accuracy of the ensemble classifier with the chosen models
        """
        
        test_set = self.test_set

        correct = 0
        total = len(test_set)

        sentiment_map = {'pos': 'positive', 'neg': 'negative', 'neu': 'neutral'}
        
        if total <= 0:
            print(f'The test set was empty')
            return 0

        # predict a label and compare to the actual label
        for item in test_set:
            text, actual_label = item
            predicted_label = self.predict_all(text)
            actual_label = sentiment_map[actual_label]
            if predicted_label == actual_label:
                correct += 1
                
        accuracy = correct / total
        print(f'ensemble accuracy: {accuracy}')
        return accuracy 
        


class ModelManager(ABC):
    """
    Abstract class representing a model manager. Subclasses should implement specific models.
    """
    
    @abstractmethod
    def preprocess_text(self, text):
        """
        Preprocess the given text.

        Args:
            text (str): The text to preprocess.

        Returns:
            Processed text in a format suitable for the specific model.
        """
        
        pass
    
    @abstractmethod
    def format_dataset(self, train_set, test_set):
        """
        Format the given train and test sets for the specific model. The exact preprocessing 
        steps may vary depending on the model.

        Args:
            train_set (list): The training dataset, where each element is a tuple of (text, label).
            test_set (list): The test dataset, similar to the training set.

        Returns:
            tuple: The formatted train_set and test_set.
        """
        
        pass
    
    @abstractmethod
    def train(self):
        """
        Train the model using the train set. The training process may vary 
        depending on the specific model.
        
        Returns:
            Model: The trained model.
        """

        pass

    @abstractmethod
    def score(self):
        """
        Score the model's performance using the test set. The scoring metric may be 
        accuracy, F1-score, etc., based on the model's implementation. Currently only supports accuracy.

        Returns:
            float: The model's score.
        """
        
        pass
    
    @abstractmethod
    def predict(self, text):
        """
        Predict the sentiment of the provided text using the model.

        Args:
            text (str): The text whose sentiment needs to be predicted.

        Returns:
            str: The predicted sentiment ('positive', 'negative', or 'neutral').
        """
        
        pass



class NaiveBayesModel(ModelManager):
    """
    Represents the Naive Bayes model for sentiment analysis. This class preprocesses the data, 
    trains the Naive Bayes classifier, and can predict sentiment for given text.

    Attributes:
        MODEL_PATH (str): Path to save or load the model.
        stop_words (set): Set of stopwords to be removed during preprocessing.
        train_set (list): Formatted list of training data.
        test_set (list): Formatted list of testing data.
        model: Trained Naive Bayes classifier.
        weight (float): The model's accuracy used as its weight in an ensemble.
    """
    
    MODEL_PATH = './models/NaiveBayes_model.joblib'
    
    def __init__(self, train_set, test_set, stop_words):

        self.stop_words = stop_words
        self.train_set, self.test_set = self.format_dataset(train_set, test_set)

        # If a saved model exists, load it. Otherwise, train it and save it
        if os.path.exists(self.MODEL_PATH):
            try:
                self.model = load(self.MODEL_PATH)
            except Exception as e:
                print(f"Error loading model from {self.MODEL_PATH}. Retraining model. Error: {e}")
        else:
            print(f"Model path {self.MODEL_PATH} does not exist. Training model.")
            self.model = self.train()
            dump(self.model, self.MODEL_PATH)
            
        self.weight = self.score()

        
    ### n-grams improved the performance of naive bayes by around 7%, but technically
    ### breaks the assumed independence of features, so it isn't a true Naive Bayes model 
    def create_ngrams(self, words, n):
        """
        Create n-grams from a list of words.

        Args:
            words (list): A list of words from which to generate n-grams.
            n (int): The size of the n-grams to generate.

        Returns:
            list: A list containing the generated n-grams.
        """
        
        return list(zip(*[words[i:] for i in range(n)]))


    def bag_of_words(self, words, ngram_range=(1, 1)):
        """
        Create a bag of words representation with specified n-gram range.

        Args:
            words (list): A list of words to be used in the bag of words representation.
            ngram_range (tuple): A tuple (min_n, max_n) specifying the range of n-gram sizes to include.
                                        Defaults to (1, 1).

        Returns:
            dict: A dictionary representing the bag of words with n-grams as keys and "True" as values.
        """

        ngram_features = {}
        for n in range(ngram_range[0], ngram_range[1] + 1):
            ngrams = self.create_ngrams(words, n)
            ngram_features.update({f"{ngram}": True for ngram in ngrams})
        return ngram_features

    
    def preprocess_text(self, text):
        """
        Manually Preprocess the text by tokenizing, removing stop words, and conversion to bag of words.

        Args:
            text (str): The text to preprocess.

        Returns:
            dict: The tokenized text in bag of words form
        """
        
        words = nltk.word_tokenize(text)

        filtered_words = [word for word in words if word.lower() not in self.stop_words]

        return self.bag_of_words(filtered_words, ngram_range=(1, 2))
        

    def format_dataset(self, train_set, test_set):
        """
        Preprocesses the train and test sets based on the Naive Bayes model requirements. 

        Args:
            train_set (list): The training dataset, where each element is a tuple of (text, label).
            test_set (list): The test dataset.
            
        Returns:
            tuple: The preprocessed train_set and test_set.
        """
        
        formatted_train_set = [(self.preprocess_text(item[0]), item[1]) for item in train_set]
        formatted_test_set = [(self.preprocess_text(item[0]), item[1]) for item in test_set]

        return formatted_train_set, formatted_test_set


    def train(self):
        """
        Train a Naive Bayes classifier using the train set.

        Returns:
            classifier: Trained Naive Bayes classifier.
        """
        
        train_set = self.train_set
        self.model = nltk.NaiveBayesClassifier.train(train_set)
        return self.model

    
    def score(self):
        """
        Computes and returns the accuracy of the trained Naive Bayes classifier on the test set.

        Prints the accuracy to standard output for reference.

        Returns:
            float: Accuracy of the Naive Bayes classifier on the test set.
        """
        
        test_set = self.test_set
        accuracy = nltk.classify.util.accuracy(self.model, test_set)
        print(f'Bayes accuracy: {accuracy}')
        return accuracy

        
    def predict(self, text):
        """
        Predicts the sentiment of a given text using the trained Naive Bayes classifier.

        Args:
            text (str): The input text for which the sentiment needs to be predicted.

        Returns:
            str: Predicted sentiment label (e.g., 'pos', 'neg', 'neu').
        """
         
        preprocessed_title_bayes = self.preprocess_text(text)
        sentiment_bayes = self.model.classify(preprocessed_title_bayes)
        return sentiment_bayes



class SVMModel(ModelManager):
    """
    Represents the Support Vector Machine (SVM) model for sentiment analysis. This class preprocesses the data, 
    trains the SVM classifier, and can predict sentiment for given text.

    Attributes:
        MODEL_PATH (str): Path to save or load the model.
        stop_words (set): Set of stopwords to be removed during preprocessing.
        train_set (list): Formatted list of training data.
        test_set (list): Formatted list of testing data.
        model: Trained SVM classifier in a pipeline.
        weight (float): The model's accuracy used as its weight in an ensemble.
    """
    
    MODEL_PATH = './models/SVM_model.joblib'

    def __init__(self, train_set, test_set, stop_words):

        self.stop_words = stop_words
        self.train_set, self.test_set = self.format_dataset(train_set, test_set)

        # If a saved model exists, load it. Otherwise, train it and save it
        if os.path.exists(self.MODEL_PATH):
            try:
                self.model = load(self.MODEL_PATH)
            except Exception as e:
                print(f"Error loading model from {self.MODEL_PATH}. Retraining model. Error: {e}")
        else:
            print(f"Model path {self.MODEL_PATH} does not exist. Training model.")
            self.model = self.train()
            dump(self.model, self.MODEL_PATH)
            
        self.weight = self.score()
        
    
    def preprocess_text(self, text):
        """
        Preprocess the given text by tokenizing and filtering stop words.
        
        Args:
            text (str): Text to be preprocessed.
        
        Returns:
            str: Preprocessed text.
        """
        
        words = nltk.word_tokenize(text)

        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        filtered_string = ' '.join(filtered_words)

        return filtered_string

        
    def format_dataset(self, train_set, test_set):
        """
        Preprocesses the train and test sets based on the SVM model requirements. 

        Args:
            train_set (list): The training dataset, where each element is a tuple of (text, label).
            test_set (list): The test dataset, similar to the training set.
            
        Returns:
            tuple: The preprocessed train_set and test_set.
        """

        # Form of a tuple consisting of text and label
        formatted_train_set = [(self.preprocess_text(item[0]), item[1]) for item in train_set]
        formatted_test_set = [(self.preprocess_text(item[0]), item[1]) for item in test_set]

        return formatted_train_set, formatted_test_set


    def train(self):
        """
        Train a Support Vector Machine (SVM) model using the traom set.

        Returns:
            pipeline: Trained SVM model in a pipeline.
        """
        
        train_set = self.train_set
        
        # SVM model requires a list of features and a list of labels
        train_features = [text for text, _ in train_set]
        train_labels = [label for _, label in train_set]
        
        self.model = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', LinearSVC(C=1,
                                 class_weight='balanced',
                                 dual=True,
                                 loss='squared_hinge',
                                 penalty='l2'))
        ])

        self.model.fit(train_features, train_labels)
        return self.model

    
    def score(self):
        """
        Computes and returns the accuracy of the trained SVM model on the test set.

        Prints the accuracy to standard output for reference.

        Returns:
            float: Accuracy of the SVM model on the test set.
        """
        
        test_set = self.test_set

        # SVM model requires a list of features and a list of labels
        test_features = [text for text, _ in test_set]
        test_labels = [label for _, label in test_set]

        accuracy = self.model.score(test_features, test_labels)
        print(f'SVM accuracy: {accuracy}')
        return accuracy

    
    def predict(self, text):
        """
        Predicts the sentiment of a given text using the trained SVM model.

        Args:
            text (str): The input text for which the sentiment needs to be predicted.

        Returns:
            str: Predicted sentiment label (e.g., 'pos', 'neg', 'neu').
        """
        
        preprocessed_title_SVM = self.preprocess_text(text)
        sentiment_SVM = self.model.predict([preprocessed_title_SVM])[0]
        return sentiment_SVM



class DistilbertModel(ModelManager):
    """
    Represents the DistilBERT model for sentiment analysis. This class preprocesses the data, 
    trains the DistilBERT classifier, and can predict sentiment for a given text.

    Attributes:
        device (torch.device): Torch device (either 'cuda' or 'cpu') used for model execution.
        label_encoder (LabelEncoder): Sklearn utility to encode and transform textual labels to integers.
        train_set (list): Formatted list of training data.
        test_set (list): Formatted list of testing data.
        stop_words (set): Set of stopwords to be removed during preprocessing.
        tokenizer: Tokenizer specific to DistilBERT model.
        model: Pre-trained or fine-tuned DistilBERT model.
        weight (float): The model's accuracy used as its weight in an ensemble.
    """
    
    def __init__(self, train_set, test_set, stop_words):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['neg', 'neu', 'pos'])
        
        self.train_set, self.test_set = self.format_dataset(train_set, test_set)
        self.stop_words = stop_words

        self.tokenizer = self.get_tokenizer() 
        self.model = self.get_model() 
        self.weight = self.score()


    def get_tokenizer(self):
        """
        Load the DistilBertTokenizer.

        Returns:
            tokenizer: The DistilBertTokenizer.
        """
        
        return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


    def get_model(self):
        """
        Load the fine-tuned DistilBERT model.

        Returns:
            model: The DistilBERT model.
        """
        
        model_path = './models/Distilbert_model'

        # If a trained distilbert model is already in the path, use it. Otherwise, train the distilbert model
        if os.path.exists(model_path):
            try:
                model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=3)
                model.to(self.device)
            except Exception as e:
                print(f"Error loading model from {model_path}. Retraining model. Error: {e}")
                model = self.train()
        else:
            print(f"Model path {model_path} does not exist. Training model.")
            model = self.train()

        return model

        
    def preprocess_text(self, text):
        """
        For DistilBERT, no preprocessing is needed as the model handles it internally.
        
        Args:
            text (str): Text to be processed.
        
        Returns:
            str: Unchanged text.
        """

        return text

        
    def format_dataset(self, train_set, test_set):
        """
        Formats and preprocesses the train and test sets suitable for the DistilBERT model.

        Args:
            train_set (list): Raw training data.
            test_set (list): Raw testing data.

        Returns:
            tuple: Formatted train_set and test_set suitable for DistilBERT model.
        """
        
        # distilbert requires number labels and dictionary format
        # label encoder also only takes in lists, so the labels must be made into a list
        formatted_train_set = [{'sentence' : item[0], 'label': self.label_encoder.transform([item[1]])[0].astype('int64')} for item in train_set]
        formatted_test_set = [{'sentence' : item[0], 'label': self.label_encoder.transform([item[1]])[0].astype('int64')} for item in test_set]

        return formatted_train_set, formatted_test_set


    def train(self):
        """
        Fine-tune a DistilBERT model using the movie reviews and Financial PhraseBank datasets.

        Returns:
            model: The fine-tuned DistilBERT model.
        """

        train_set = self.train_set

        # Create the model and move it to the appropriate device
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
        model.to(self.device)  

        # Convert datasets to format for DistilBERT
        train_set = [{'input_ids': self.tokenizer.encode(item['sentence'], truncation=True, max_length=512, padding='max_length'),
                      'attention_mask': self.tokenizer.encode(item['sentence'], truncation=True, max_length=512, padding='max_length'),
                      'labels': item['label']} for item in train_set]

        # Convert lists to tensors and move them to the appropriate device
        train_dataset = [{'input_ids': torch.tensor(item['input_ids']).squeeze().to(self.device),
                          'attention_mask': torch.tensor(item['attention_mask']).squeeze().to(self.device),
                          'labels': torch.tensor(item['labels']).to(self.device)} for item in train_set]
        

        # Define the training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        # Create the Trainer and train the model
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()

        model.save_pretrained('./models/Distilbert_model')
        return model

  
    def score(self):
        """
        Measure the accuracy of the DistilBERT model on the test set.

        Returns:
            float: The accuracy of the model.
        """
        
        test_set = self.test_set
        
        metric = evaluate.load("accuracy")

        # Convert datasets to format for DistilBERT
        test_set = [{'input_ids': self.tokenizer.encode(item['sentence'], truncation=True, max_length=512, padding='max_length'),
                      'attention_mask': self.tokenizer.encode(item['sentence'], truncation=True, max_length=512, padding='max_length'),
                      'labels': item['label']} for item in test_set]
        
        # Convert lists to tensors
        test_dataset = [{'input_ids': torch.tensor(item['input_ids']).squeeze(),
                          'attention_mask': torch.tensor(item['attention_mask']).squeeze(),
                          'labels': torch.tensor(item['labels'])} for item in test_set]

        test_dataloader = DataLoader(test_dataset, batch_size=38)

        # Handle the dataset in batches to avoid using too much memory using the dataloader
        for batch in test_dataloader:
            # Prepare input tensors for model
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Make predictions
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            metric.add_batch(predictions=predictions, references=labels)

        # Calculate the accuracy
        accuracy_dict = metric.compute()
        accuracy = accuracy_dict['accuracy']
        print(f'DistilBERT accuracy: {accuracy}')
        return accuracy

    
    def predict(self, text):
        """
        Classify the sentiment of a text using the fine-tuned DistilBERT model.
        
        Args:
            text (str): The text to classify.

        Returns:
            str: The classified sentiment ('positive', 'negative', or 'neutral').
        """
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        sentiment = torch.argmax(logits, dim=1).item()

        if sentiment == 0:
            return 'neg'
        elif sentiment == 2:
            return 'pos'
        else:
            return 'neu'



class LogisticRegressionModel(ModelManager):
    """
    Represents LogisticRegression model for sentiment analysis. This class preprocesses the data, 
    trains the LogisticRegression classifier, and can predict sentiment for given text.

    Attributes:
        MODEL_PATH (str): Path to save or load the model.
        stop_words (set): Set of stopwords to be removed during preprocessing.
        train_set (list): Formatted list of training data.
        test_set (list): Formatted list of testing data.
        model: Trained SVM classifier in a pipeline.
        weight (float): The model's accuracy used as its weight in an ensemble.
    """

    MODEL_PATH = './models/LogisticRegression_model.joblib'

    def __init__(self, train_set, test_set, stop_words):

        self.stop_words = stop_words
        self.train_set, self.test_set = self.format_dataset(train_set, test_set)

        # If a saved model exists, load it. Otherwise, train it and save it
        if os.path.exists(self.MODEL_PATH):
            try:
                self.model = load(self.MODEL_PATH)
            except Exception as e:
                print(f"Error loading model from {self.MODEL_PATH}. Retraining model. Error: {e}")
        else:
            print(f"Model path {self.MODEL_PATH} does not exist. Training model.")
            self.model = self.train()
            dump(self.model, self.MODEL_PATH)

        self.weight = self.score()

        
    def preprocess_text(self, text):
        """
        Preprocess the given text by tokenizing and filtering stop words.
        
        Args:
            text (str): Text to be preprocessed.
        
        Returns:
            str: Preprocessed text.
        """

        words = nltk.word_tokenize(text)

        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        filtered_string = ' '.join(filtered_words)

        return filtered_string

    
    def format_dataset(self, train_set, test_set):
        """
        Preprocesses the train and test sets based on the LogisticRegression model requirements. 

        Args:
            train_set (list): The training dataset, where each element is a tuple of (text, label).
            test_set (list): The test dataset, similar to the training set.
            
        Returns:
            tuple: The preprocessed train_set and test_set.
        """
        
        # Form of a tuple consisting of text and label
        formatted_train_set = [(self.preprocess_text(item[0]), item[1]) for item in train_set]
        formatted_test_set = [(self.preprocess_text(item[0]), item[1]) for item in test_set]

        return formatted_train_set, formatted_test_set

    
    def train(self):
        """
        Train a LogisticRegression model using the train set.

        Returns:
            pipeline: Trained LogisticRegression model in a pipeline.
        """

        train_set = self.train_set

        # LogisticRegression model requires a list of features and a list of labels
        train_features = [text for text, _ in train_set]
        train_labels = [label for _, label in train_set]

        self.model = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', LogisticRegression(C= 1,
                                          penalty= 'l2',
                                          solver= 'saga',
                                          class_weight= 'balanced',
                                          fit_intercept= True,
                                          ))
        ])
        
        self.model.fit(train_features, train_labels)
        return self.model

    
    def score(self):
        """
        Measure the accuracy of the LogisticRegression model on the test set.

        Returns:
            float: The accuracy of the model.
        """

        test_set = self.test_set

        # Logistic Regression model requires a list of features and a list of labels
        test_features = [text for text, _ in test_set]
        test_labels = [label for _, label in test_set]

        accuracy = self.model.score(test_features, test_labels)
        print(f'LogisticRegression accuracy: {accuracy}')
        return accuracy

    
    def predict(self, text):
        """
        Predicts the sentiment of a given text using the trained LogisticRegression model.

        Args:
            text (str): The input text for which the sentiment needs to be predicted.

        Returns:
            str: Predicted sentiment label (e.g., 'pos', 'neg', 'neu').
        """

        preprocessed_text_LR = self.preprocess_text(text)
        prediction = self.model.predict([preprocessed_text_LR])[0]
        return prediction



class RandomForestModel(ModelManager):
    """
    Represents RandomForest model for sentiment analysis. This class preprocesses the data, 
    trains the RandomForest classifier, and can predict sentiment for given text.

    Attributes:
        MODEL_PATH (str): Path to save or load the model.
        stop_words (set): Set of stopwords to be removed during preprocessing.
        train_set (list): Formatted list of training data.
        test_set (list): Formatted list of testing data.
        model: Trained SVM classifier in a pipeline.
        weight (float): The model's accuracy used as its weight in an ensemble.
    """

    MODEL_PATH = './models/RandomForest_model.joblib'

    def __init__(self, train_set, test_set, stop_words):

        self.stop_words = stop_words
        self.train_set, self.test_set = self.format_dataset(train_set, test_set)

        # If a saved model exists, load it. Otherwise, train it and save it
        if os.path.exists(self.MODEL_PATH):
            try:
                self.model = load(self.MODEL_PATH)
            except Exception as e:
                print(f"Error loading model from {self.MODEL_PATH}. Retraining model. Error: {e}")
        else:
            print(f"Model path {self.MODEL_PATH} does not exist. Training model.")
            self.model = self.train()
            dump(self.model, self.MODEL_PATH)
            
        self.weight = self.score()

        
    def preprocess_text(self, text):
        """
        Preprocess the given text by tokenizing and filtering stop words.
        
        Args:
            text (str): Text to be preprocessed.
        
        Returns:
            str: Preprocessed text.
        """

        words = nltk.word_tokenize(text)

        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        filtered_string = ' '.join(filtered_words)

        return filtered_string

    
    def format_dataset(self, train_set, test_set):
        """
        Preprocesses the train and test sets based on the RandomForest model requirements. 

        Args:
            train_set (list): The training dataset, where each element is a tuple of (text, label).
            test_set (list): The test dataset, similar to the training set.
            
        Returns:
            tuple: The preprocessed train_set and test_set.
        """

        # Form of a tuple consisting of text and label
        formatted_train_set = [(self.preprocess_text(item[0]), item[1]) for item in train_set]
        formatted_test_set = [(self.preprocess_text(item[0]), item[1]) for item in test_set]

        return formatted_train_set, formatted_test_set

    
    def train(self):
        """
        Train a RandomForest model using the train set.

        Returns:
            pipeline: Trained RandomForest model in a pipeline.
        """
         
        train_set = self.train_set

        # RandomForest model requires a list of features and a list of labels
        train_features = [text for text, _ in train_set]
        train_labels = [label for _, label in train_set]

        self.model = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', RandomForestClassifier(max_depth= None,
                                              bootstrap= False,
                                              class_weight = 'balanced',
                                              criterion= 'gini',
                                              min_samples_leaf= 1,
                                              n_estimators= 100,
                                              min_samples_split= 10,
                                              max_features= 'sqrt'))
        ])
        
        self.model.fit(train_features, train_labels)
        return self.model

    
    def score(self):
        """
        Measure the accuracy of the RandomForest model on the test set.

        Returns:
            float: The accuracy of the model.
        """

        test_set = self.test_set

        # RandomForest model requires a list of features and a list of labels
        test_features = [text for text, _ in test_set]
        test_labels = [label for _, label in test_set]

        accuracy = self.model.score(test_features, test_labels)
        print(f'RandomForest accuracy: {accuracy}')
        return accuracy

    
    def predict(self, text):
        """
        Predicts the sentiment of a given text using the trained RandomForest model.

        Args:
            text (str): The input text for which the sentiment needs to be predicted.

        Returns:
            str: Predicted sentiment label (e.g., 'pos', 'neg', 'neu').
        """

        preprocessed_text_RF = self.preprocess_text(text)
        prediction = self.model.predict([preprocessed_text_RF])[0]
        return prediction


def other_datasets_load():
    """
    This is a combination of other datasets that can be used for the model that were not used.
    The custom dataset does much better in allowing for generalization to real world data than these.
    """

    # Ensure the same seed is used for all random operations to prevent leakage when saving models
    np.random.seed(42)
        
    # Create a map to convert the integer labels from the financial dataset to word labels 
    label_map = {0: 'neg', 1: 'neu', 2: 'pos'}
        
    # Load the datasets
    financial_dataset = load_dataset('financial_phrasebank', 'sentences_allagree')   
    movie_reviews_data = [{'sentence': ' '.join(movie_reviews.words(fileid)), 'label': category}
                  for category in movie_reviews.categories()
                  for fileid in movie_reviews.fileids(category)]

    # Original text and labels without any pre-processing in tuple form
    movie_reviews_data = [(item['sentence'], item['label']) for item in movie_reviews_data]
    financial_data = [(item['sentence'], label_map[item['label']]) for item in financial_dataset['train']]

    # Sample a subset
    subset_size = int(len(movie_reviews_data) * 0.60)
    indices = np.random.choice(len(movie_reviews_data), size=subset_size, replace=False)
    movie_reviews_data_subset = [movie_reviews_data[i] for i in indices]

    # Combine the datasets
    dataset = financial_data + movie_reviews_data_subset

    # Shuffle the dataset
    shuffled_indices = np.random.permutation(len(dataset))
    shuffled_dataset = [dataset[i] for i in shuffled_indices]

    train_set, test_set = train_test_split(dataset, test_size=0.25, shuffle=False)

    return train_set, test_set

def download_nltk_resources():
    """Download necessary NLTK resources if not already available."""

    resources = ['punkt', 'stopwords', 'movie_reviews']
    
    for resource in resources:
        nltk.download(resource)

download_nltk_resources()
