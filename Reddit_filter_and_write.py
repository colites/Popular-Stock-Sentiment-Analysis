import csv
import datetime
import re
import psycopg2
import psycopg2.extras
import Web_Scrapper_Reddit as Rscrap
import database_config
import Evaluation_model as eval
import SQL as dat

## Checks whether there is a stock ticker mentioned
## Returns a bool and the word if it is a possible ticker
def Mentioned_Potential_Ticker(title):    
    words = title.split()
    
    ##check each word for $ or for all Uppercase when word meets ticker length reqs,
    ## Further checks potential tickers without $
    for word in words:
        if '$' in word and word.isupper():
            filter_Tickers(word)
            return True, word
        if word.isupper() and 1 < len(word) < 5:
            blacklisted, word = filter_Tickers(word)

            ## check the length again because removing symbols shortens the word also check if word is in the blacklist
            if not blacklisted and 1 < len(word):
                word = '$' + word
                return True, word
    return False, "None"

## Creates a new ticker that replaces any non_word characters
def filter_Ticker_Characters(ticker):
    ticker = re.sub("[^a-zA-Z]+", "", ticker)
    return ticker

## Further filters out tickers by checking the blacklist
## returns bool confirming blacklist status and the filtered ticker 
def filter_Tickers(ticker):
    BLACKLIST = ['CEO','EV','YOLO','API', 'EU', 'IQ','POV','PUTS','PUT','CALL','SELL', 'BUY', 'ETF','SPAC','WSB', 'USA', 'US', 'IRA', 'PDF','UK', 'NFT', 'CPI', 'NO','OR','HODL']

    ticker = filter_Ticker_Characters(ticker)
    if ticker in BLACKLIST:
        return True, ticker

    return False, ticker

def main(amount):
    ##make a connection to a database and a cursor to execute queries
    connection = psycopg2.connect(host = database_config.DB_HOST, database = database_config.DB_NAME, user = database_config.DB_USER, password = database_config.DB_PASS)
    connection.autocommit = True
    cursor = connection.cursor(cursor_factory = psycopg2.extras.DictCursor)

    ## Create a CSV file with UTF-8 Encoding
    with open('Reddit-Titles-new.csv', 'w', encoding='UTF-8') as csvfile:
        file = csv.writer(csvfile)
        reader = csv.reader(csvfile)
        file.writerow(['Ticker', 'Post_text', 'Post_Url', 'Sentiment'])

        dat.create_database(connection, cursor)

        ## Connect to the newly created database
        connection = psycopg2.connect(host = database_config.DB_HOST, database = 'hype_stock', user = database_config.DB_USER, password = database_config.DB_PASS)
        connection.autocommit = True
        cursor = connection.cursor(cursor_factory = psycopg2.extras.DictCursor)

        ## create tables for the database if they do not exist
        dat.create_tables(connection, cursor)
        
        mention = {}
        pmention = {}
        nmention = {}
        neutral = {}
        tickerset = set()
        submissions = Rscrap.Get_Submissions_Any(amount, 'wallstreetbets')

        ## Go through each submission, filter it, add it to the tables, and then classify it
        for submission in submissions:
            title = submission[0]
            url = submission[1]
            author = submission[2]

            boole, ticker_word = Mentioned_Potential_Ticker(title)
            if boole:
                try:
                    time, low, high = Rscrap.Get_stock_data(ticker_word)
                    if low == None or high == None:
                        print("non-existent stock")
                        continue
                    stock_data = (time, ticker_word, low, high)
                    dat.insert_stock(connection, cursor, stock_data)

                    sentiment = eval.Eval_text(title)

                    ## add sentiment statistics to dictionaries for future visualizations
                    mention[ticker_word] = mention.get(ticker_word,0) + 1
                    if sentiment == 'pos':
                        pmention[ticker_word] = pmention.get(ticker_word,0) + 1
                        sentiment = 'positive' 
                    elif sentiment == 'neg':
                        nmention[ticker_word] = nmention.get(ticker_word,0) + 1
                        sentiment = 'negative' 
                    else:
                        neutral[ticker_word] = neutral.get(ticker_word,0) + 1
                        sentiment = 'neutral'

                    dat.insert_mention(connection, cursor, stock_data, sentiment, title, url)
                    file.writerow([str(ticker_word), title, url, sentiment])
                except Exception as e:
                    print(ticker_word)
                    print("error:", e)
                    print("wrong")
                    
    ## new csv file for stats of each stock
    with open('Reddit-Stock-Stats.csv', 'w', encoding='UTF-8') as csvfile:
        stats = csv.writer(csvfile)
        stats.writerow(['ticker', 'total_mentions', 'psentiment', 'nsentiment'])
        tickers_in_database = list(mention.keys())
        for index in range(len(tickers_in_database)):
            ticker = tickers_in_database[index]
            number_of_positive = pmention[ticker] if ticker in pmention else 0
            number_of_negative = nmention[ticker] if ticker in nmention else 0
            stats.writerow([ticker, mention[ticker], number_of_positive, number_of_negative])

    connection.close()
