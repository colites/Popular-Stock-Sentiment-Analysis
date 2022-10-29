import csv
import Web_Scrapper_Reddit as Rscrap
import database_config
import datetime
import psycopg2
import psycopg2.extras
import alpaca_trade_api as tradeapi
import Evaluation_model.py as eval

## Checks whether there is a stock ticker mentioned
## Returns a bool and the word if it is a possible ticker
def Mentioned_Potential_Ticker(title):    
    words = title.split()
    
    ##check each word for $ or for all Uppercase when word is greater than 1 but less than 5,
    ## in accordance with how stock tickers are listed on exchanges. Further checks potential tickers without $
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

## Creates a new ticker that replaces any non_word characters that were part of the tickers.
## returns the ticker without the unneeded characters
def filter_Ticker_Characters(ticker):
    if ':' in ticker:
        ticker = ticker.replace(":", "")
    if '?' in ticker:
        ticker = ticker.replace("?", "")
    if '.' in ticker:
        ticker = ticker.replace(".", "")
    if ',' in ticker:
        ticker = ticker.replace(",", "")
    if '$' in ticker:
        ticker = ticker.replace("$", "")
    if '!' in ticker:
        ticker = ticker.replace("!", "")
    return ticker

## Further filters out tickers by checking if the possible ticker
## corresponds to a word on the blacklist, which is not a ticker.
## returns bool confirming blacklist status and the filtered ticker 
def filter_Tickers(ticker):
    BLACKLIST = ['EV','YOLO','API', 'EU', 'IQ','POV','PUTS','PUT','CALL','SELL', 'BUY', 'ETF','SPAC','WSB', 'USA', 'US', 'IRA', 'PDF','UK', 'NFT', 'CPI', 'NO','OR','HODL']

    ticker = filter_Ticker_Characters(ticker)
    if ticker in BLACKLIST:
        return True, ticker

    return False, ticker

connection = psycopg2.connect(host = database_config.DB_HOST, database = database_config.DB_NAME, user = database_config.DB_USER, password = database_config.DB_PASS)

#### make a dictionary cursor to loop through query results as a dictionary
cursor = connection.cursor(cursor_factory = psycopg2.extras.DictCursor)

api = tradeapi.REST(database_config.API_KEY, database_config.API_SECRET, base_url=database_config.API_URL)

assets = api.list_assets()

for asset in assets:
    cursor.execute("""
        INSERT INTO stock (name, symbol) 
        VALUES (%s, %s)
    """, (asset.name, asset.symbol))

connection.commit()

cursor.execute("""
    SELECT * FROM stock
""")
rows = cursor.fetchall()

#### select the stock ticker using the $ followed by the symbol
stocks = {}
for row in rows: 
    stocks['$' + row['symbol']] = row['id']

## Create a CSV file with UTF-8 Encoding
file = csv.writer(open('Reddit-Titles.csv', 'w',encoding='UTF-8'))
file.writerow(['Title', 'Ticker Mention', 'Ticker Name'])

mention = {}
pmention = {}
nmention = {}
neutral = {}
tickerset = set()
submissions = Rscrap.Get_Submissions_Any(100, 'wallstreetbets')

for submission in submissions:
    title = submission[0]
    url = submission[1]
    author = submission[2]

    bool, ticker_word = Mentioned_Potential_Ticker(title)
    
    if bool:
        try:
            cursor.execute("""
                INSERT INTO mention (stock_id, message, source, url)
                VALUES (%s, %s, 'wallstreetbets', %s)""", (stocks[ticker_word], title, url))
            connection.commit()
            mention[ticker_word] = mention.get(tickerword,0) + 1
            if eval.classifier() == True:
                pmention[ticker_word] = pmention.get(tickerword,0) + 1
            elif eval.classifier() == False:
                nmention[ticker_word] = nmention.get(tickerword,0) + 1
            else:
                neutral[ticker_word] = neutral.get(tickerword,0) + 1
        except Exception as e:
            print(e)
            connection.rollback()
        file.writerow([str(ticker_word), title, url])
