import csv
import re
import psycopg2
import psycopg2.extras
import Web_Scrapper_Reddit as Rscrap
import database_config
import Evaluation_model as eval
import SQL as dat

BLACKLIST = ['CEO', 'CFO', 'CTO', 'COO','FDIC', 'AI', 'EV', 'YOLO', 'JPOW', 'YES', 'API', 'EU', 'UAE', 'IQ','POV','PUTS','PUT','CALL','SELL', 'BUY', 'ETF','SPAC','WSB', 'USA', 'US', 'IRA','IRS', 'PDF', 'UK', 'NFT', 'CPI', 'NO','OR','HODL', 'FOMO']

def mentioned_potential_ticker(title):
    """
    Check if there's a potential stock ticker mentioned in the title.
    
    Args:
        title (str): The post title.
    
    Returns:
        tuple: A tuple containing a boolean and the potential stock ticker.
    """
    
    words = title.split()
    
    for word in words:
        if word.isupper() and 1 < len(word) < 5:
            blacklisted, ticker = filter_tickers(word)
            
            # check the blacklist and also the length due to filtering shortening the word
            if not blacklisted and 1 < len(ticker):
                return True, f'${ticker}'
            
    return False, "None"


def filter_ticker_characters(ticker):
    """
    Remove any non-word characters from the ticker.
    
    Args:
        ticker (str): The stock ticker.
    
    Returns:
        str: The filtered stock ticker.
    """
    
    return re.sub(r'[^a-zA-Z]', '', ticker)


def filter_tickers(ticker):
    """
    Filter out blacklisted tickers.
    
    Args:
        ticker (str): The stock ticker.
    
    Returns:
        tuple: A tuple containing a boolean for the blacklist status and the filtered ticker.
    """
    
    ticker = filter_ticker_characters(ticker)
    if ticker in BLACKLIST:
        return True, ticker

    return False, ticker


def reddit_eval(connection, cursor, submission, analyzer, mentions, file):
    """
    Analyzes a Reddit submission for potential stock mentions and evaluates sentiments.

    Args:
        connection: connection object.
        cursor: cursor object for executing SQL queries.
        submission (tuple): Tuple containing Reddit submission data. Expected format: (title, url, author, date).
        analyzer: Sentiment analysis object.
        mentions (dict): Dictionary tracking stock mentions and sentiments.
        file (csv.writer): CSV writer object to record stock mentions and sentiments.

    Returns:
        None
    """
    
    title = submission[0]
    url = submission[1]
    author = submission[2]
    date = submission[3]
    
    boole, ticker_word = mentioned_potential_ticker(title)
    if boole:
        try:
            time, low, high = Rscrap.Get_stock_data(ticker_word)
            if low is None or high is None:
                print(f"non-existent stock")
                return
            stock_data = (time, ticker_word, low, high)
            dat.insert_stock(connection, cursor, stock_data)

            sentiment = analyzer.Eval_text(title)

            # Update sentiment statistics in the mentions dictionary
            if ticker_word not in mentions:
                mentions[ticker_word] = {'total_mentions': 0, 'positive': 0, 'negative': 0, 'neutral': 0}

            mentions[ticker_word]['total_mentions'] += 1
            mentions[ticker_word][sentiment] += 1

            dat.insert_mention_and_sentiment(connection, cursor, stock_data, sentiment, title, date)
            file.writerow([str(ticker_word), title, url, sentiment])
        except Exception as e:
            print(ticker_word)
            print("error:", e)
            return

def financial_news_eval(connection, cursor, article, analyzer, mentions, file, ticker_word):
    """
    Does sentiment analysis on a financial news article.

    Args:
        connection: connection object.
        cursor: cursor object for executing SQL queries.
        article (tuple): Tuple containing article data. Expected format: (title, url, author, date).
        analyzer: Sentiment analysis object.
        mentions (dict): Dictionary tracking stock mentions and sentiments.
        file (csv.writer): CSV writer object to record stock mentions and sentiments.
        ticker_word (str): The ticker symbol associated with the news article.

    Returns:
        None
    """
    
    title = article[0]
    url = article[1]
    author = article[2]
    date = article[3]

    try:
        time, low, high = Rscrap.Get_stock_data(ticker_word)
        if low is None or high is None:
            print(f"non-existent stock")
            return
        stock_data = (time, ticker_word, low, high)
        dat.insert_stock(connection, cursor, stock_data)

        sentiment = analyzer.Eval_text(title)

        # Update sentiment statistics in the mentions dictionary
        if ticker_word not in mentions:
            mentions[ticker_word] = {'total_mentions': 0, 'positive': 0, 'negative': 0, 'neutral': 0}

        mentions[ticker_word]['total_mentions'] += 1
        mentions[ticker_word][sentiment] += 1

        dat.insert_mention_and_sentiment(connection, cursor, stock_data, sentiment, title, date)
        file.writerow([str(ticker_word), title, url, sentiment])
    except Exception as e:
        print(ticker_word)
        print("error:", e)
        return
        
def main_pipeline(num_posts, subreddit, models):
    """
    Main function to run the pipeline for web scraping, sentiment analysis, and storing data.

    Args:
        num_posts (int): The number of submissions to fetch from the subreddit.
        subreddit (str): The name of the subreddit from which to fetch submissions.
        models (list): a list of strings that contains the models to use for sentiment analysis.
        
    """
     
    # Make a connection to a database and a cursor to execute queries
    # Autocommit is needed to create a database in python
    connection1 = psycopg2.connect(host = database_config.DB_HOST, database = database_config.DB_NAME, user = database_config.DB_USER, password = database_config.DB_PASS)
    connection1.autocommit = True
    cursor1 = connection1.cursor(cursor_factory = psycopg2.extras.DictCursor)

    # Create a CSV file with UTF-8 Encoding
    with open('Reddit-Titles-new.csv', 'w', encoding='UTF-8') as csvfile:
        file = csv.writer(csvfile)
        file.writerow(['Ticker', 'Post_text', 'Post_Url', 'Sentiment'])

        connection, cursor = dat.create_database(connection1, cursor1)

        # Create tables for the database if they do not exist
        dat.create_tables(connection, cursor)

        mentions = {}
        
        submissions = Rscrap.Get_Submissions_Any(num_posts, subreddit)
        analyzer = eval.SentimentAnalyzer(classifiers=models)

        # Do sentiment eval on the reddit titles
        for submission in submissions:
            reddit_eval(connection, cursor, submission, analyzer, mentions, file)

        # Do sentiment eval on the financial news texts with the tickers gathered on reddit
        for ticker_word in list(mentions.keys()):
            ticker_word = filter_ticker_characters(ticker_word)
            ticker_news = Rscrap.get_financial_news(ticker_word)
            ticker_word = '$' + ticker_word
            for article in ticker_news:
                financial_news_eval(connection, cursor, article, analyzer, mentions, file, ticker_word)
    
    # Write the stock stats to a new CSV file                    
    with open('Reddit-Stock-Stats.csv', 'w', encoding='UTF-8') as csvfile:
        stats_writer = csv.writer(csvfile)
        stats_writer.writerow(['ticker', 'total_mentions', 'positive', 'negative', 'neutral'])

        for ticker, stats in mentions.items():
            stats_writer.writerow([ticker, stats['total_mentions'], stats['positive'], stats['negative'], stats['neutral']])

    cursor.close()
    connection.close()
