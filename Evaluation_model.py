import functools
import os
import gc
import random
import nltk
import nltk.classify.util, nltk.metrics
from nltk.corpus import movie_reviews, stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer, DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import Dataset, DataLoader


class SentimentAnalyzer:

    def __init__(self, classifiers=['bayes','SVM', 'distilbert']):
            
        self.classifiers = classifiers
        self.stop_words = set(stopwords.words('english'))
        self.bayes_classifier = self.train_NaiveBayes_Classifier() if 'bayes' in classifiers else None
        self.SVM_classifier = self.train_SVM_Classifier() if 'SVM' in classifiers else None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['neg', 'neu', 'pos'])
        self.tokenizer = self.get_tokenizer() if 'distilbert' in classifiers else None
        self.model = self.get_model() if 'distilbert' in classifiers else None

        # Initial weights for each classifier
        self.bayes_weight = 1.0
        self.svm_weight = 1.0
        self.distilbert_weight = self.get_distilbert_accuracy() if 'distilbert' in classifiers else None


    ### n-grams improved the performance of naive bayes by around 10%
    def create_ngrams(self, words, n):
        """
        Create n-grams from a list of words.

        Args:
            words (list): A list of words from which to generate n-grams.
            n (int): The size of the n-grams to generate.

        Returns:
            zip object: An iterable containing the generated n-grams as tuples.
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


    def preprocess_title(self, title, classifier):
        """
        Preprocess the text by tokenizing, removing stop words, and creating a bag of words representation.

        Args:
            title (str): The text to preprocess.
            classifier (str): The type of classifier to preprocess for ('bayes' or 'SVM').

        Returns:
            (dict or str): The tokenized title depending on the classifier.
        """
        
        words = nltk.word_tokenize(title)

        filtered_words = [word for word in words if word.lower() not in self.stop_words]

        if classifier == 'bayes':
            return self.bag_of_words(filtered_words)
        elif classifier == 'SVM':
            filtered_string = ' '.join(filtered_words)
            return filtered_string
        else:
            return f"not a classifier"

        
    def load_datasets(self, classifier):
        """
        Load the movie reviews and Financial PhraseBank datasets, preprocess them for the specified classifier, 
        then create the train and test sets.
        
        Args:
            classifier (str): The type of classifier to preprocess for ('bayes' or 'SVM').

        Returns:
            tuple: A tuple containing the train_set and test_set.
        """

        financial_dataset = load_dataset('financial_phrasebank', 'sentences_allagree')

        # Create a map to convert the integer labels from the financial dataset to word labels 
        label_map = {0: 'neg', 1: 'neu', 2: 'pos'}

        financial_data = [(self.preprocess_title(item['sentence'], classifier), label_map[item['label']]) 
                              for item in financial_dataset['train']]

        # Format the datasets based on the required format for each model
        if classifier == 'SVM':
            movie_reviews_data = [(self.preprocess_title(' '.join(movie_reviews.words(fileid)), classifier), category)
                                   for category in movie_reviews.categories()
                                   for fileid in movie_reviews.fileids(category)]
        
        elif classifier == 'bayes':
            negativeIds = movie_reviews.fileids('neg')
            positiveIds = movie_reviews.fileids('pos')
        
            negfeats = [(self.bag_of_words(movie_reviews.words(fileids=[f]), ngram_range=(1, 2)), 'neg') for f in negativeIds]
            posfeats = [(self.bag_of_words(movie_reviews.words(fileids=[f]), ngram_range=(1, 2)), 'pos') for f in positiveIds]

            movie_reviews_data = negfeats + posfeats


        elif classifier == 'distilbert':
            movie_reviews_data = [{'sentence': ' '.join(movie_reviews.words(fileid)), 'label': category}
                      for category in movie_reviews.categories()
                      for fileid in movie_reviews.fileids(category)]
            
            # distilbert requires number labels
            movie_reviews_data = [{'sentence': item['sentence'], 'label': self.label_encoder.transform([item['label']])[0]} for item in movie_reviews_data]

            financial_data = [{'sentence': item['sentence'], 'label': item['label']} for item in financial_dataset['train']]
           

        subset_size = int(len(movie_reviews_data) * 0.5)
        movie_reviews_data_subset = random.sample(movie_reviews_data, subset_size)

        dataset = financial_data + movie_reviews_data_subset
        random.shuffle(dataset)
        train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
        return train_set, test_set
    
    
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
        
        model_path = './model'

        # If a trained distilbert model is already in the path, use it. Otherwise, train the distilbert model
        if os.path.exists(model_path):
            try:
                model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=3)
                model.to(self.device)
            except Exception as e:
                print(f"Error loading model from {model_path}. Retraining model. Error: {e}")
                model = self.train_DistilBert_Classifier()
        else:
            print(f"Model path {model_path} does not exist. Training model.")
            model = self.train_DistilBert_Classifier()

        return model

    
    def train_DistilBert_Classifier(self):
        """
        Fine-tune a DistilBERT model using the movie reviews and Financial PhraseBank datasets.

        Returns:
            model: The fine-tuned DistilBERT model.
        """

        train_set, _ = self.load_datasets('distilbert')

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

        model.save_pretrained('./model')
        return model


    def train_NaiveBayes_Classifier(self):
        """
        Train a Naive Bayes classifier using the movie reviews dataset.

        Returns:
            classifier: Trained Naive Bayes classifier.
        """
        
        train_set, test_set = self.load_datasets('bayes')
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        accuracy = nltk.classify.util.accuracy(classifier, test_set)
        self.bayes_weight = accuracy
        print(f'Bayes accuracy: {accuracy}')
        return classifier


    def train_SVM_Classifier(self):
        """
        Train a Support Vector Machine (SVM) classifier using the movie reviews dataset.

        Returns:
            pipeline: Trained SVM classifier in a pipeline.
        """
        
        train_set, test_set = self.load_datasets('SVM')
        
        ## SVM classifier requires a list of features and a list of labels
        train_features = [text for text, _ in train_set]
        train_labels = [label for _, label in train_set]
        test_features = [text for text, _ in test_set]
        test_labels = [label for _, label in test_set]

        pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', LinearSVC())
        ])

        pipeline.fit(train_features, train_labels)
        accuracy = pipeline.score(test_features, test_labels)
        self.svm_weight = accuracy
        print(f'SVM accuracy: {accuracy}')
        return pipeline


    def distilbert_DeepLearning_Classifier(self, text):
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


    def get_distilbert_accuracy(self):
        """
        Measure the accuracy of the DistilBERT model on the combined dataset.

        Returns:
            float: The accuracy of the model.
        """
        
        _, test_set = self.load_datasets('distilbert')

        metric = load_metric("accuracy")

        # Convert datasets to format for DistilBERT
        test_set = [{'input_ids': self.tokenizer.encode(item['sentence'], truncation=True, max_length=512, padding='max_length'),
                      'attention_mask': self.tokenizer.encode(item['sentence'], truncation=True, max_length=512, padding='max_length'),
                      'labels': item['label']} for item in test_set]

        true_labels = [item['labels'] for item in test_set]
        
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

            
    def Eval_text(self, title):
        """
        Evaluate text using the chosen classifiers (Naive Bayes, SVM, and DistilBERT) and return the majority sentiment.

        Args:
            title (str): The text to evaluate.

        Returns:
            str: The majority sentiment ('positive', 'negative', or 'neutral').
        """

        sentiment_map = {'pos': 'positive', 'neg': 'negative', 'neu': 'neutral'}
        sentiment_count = {'positive': 0, 'negative': 0, 'neutral': 0}

        # If the model is chosen, evaluate the text and vote based on the weight of the model
        if 'bayes' in self.classifiers:
            preprocessed_title_bayes = self.preprocess_title(title,'bayes')
            sentiment_bayes = self.bayes_classifier.classify(preprocessed_title_bayes)
            sentiment_bayes = sentiment_map[sentiment_bayes]
            sentiment_count[sentiment_bayes] += self.bayes_weight

        if 'SVM' in self.classifiers:
            preprocessed_title_SVM = self.preprocess_title(title,'SVM')
            sentiment_SVM = self.SVM_classifier.predict([preprocessed_title_SVM])[0]
            sentiment_SVM = sentiment_map[sentiment_SVM]
            sentiment_count[sentiment_SVM] += self.svm_weight

        if 'distilbert' in self.classifiers:
            sentiment_distilbert = self.distilbert_DeepLearning_Classifier(title)
            sentiment_distilbert = sentiment_map[sentiment_distilbert]
            sentiment_count[sentiment_distilbert] += self.distilbert_weight

        # Return the sentiment with the highest count
        return max(sentiment_count, key=sentiment_count.get)

 
def download_nltk_resources():
    """Download necessary NLTK resources if not already available."""

    resources = ['punkt', 'stopwords', 'movie_reviews']
    
    for resource in resources:
        nltk.download(resource)

download_nltk_resources()
