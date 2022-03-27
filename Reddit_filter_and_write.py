import csv
import Web_Scrapper_Reddit as Rscrap

## Create the CSV file with UTF-8 Encoding
file = csv.writer(open('Reddit-Titles.csv', 'w',encoding='UTF-8'))
file.writerow(['Title', 'Ticker Mention', 'Ticker Name'])

tickers = set()
titles, urls, authors = Rscrap.Get_Submissions_Any(50, 'wallstreetbets')

## Checks whether there is a stock ticker mentioned
## Returns a bool and the word if it is a possible ticker
def Mentioned_Potential_Ticker(title):    
    words = title.split()
    
    ##check each word for $ or for all Uppercase when word is greater than 1 but less than 5,
    ## in accordance with how stock tickers are listed on exchanges. Further checks potential tickers without $
    for word in words:
        if '$' in word and word.isupper():
            filter_Tickers(word, tickers)
            return True, word
        if word.isupper() and 1 < len(word) < 5:
            bool = filter_Tickers(word,tickers)
            if bool:
                return True, word
    return False, "None"

## Creates a new ticker that replaces any non_word characters that were part of the tickers.
## returns the ticker without the unneeded characters
def filter_Ticker_Characters(ticker):
    if ':' in ticker:
        ticker = ticker.replace(":", "")
    if '?' in ticker:
        ticker = ticker.replace("?", "")
    if '$' in ticker:
        ticker = ticker.replace("$", "")
    if '.' in ticker:
        ticker = ticker.replace(".", "")
    if ',' in ticker:
        ticker = ticker.replace(",", "")
    return ticker

## Further filters out tickers by checking if the ticker is already mentioned in another post or if the possible ticker
## corresponds to a word on the blacklist, which is not a ticker.
## returns bool confirming ticker status 
## also adds the tickers to the ticker set that will be used to make the database.
def filter_Tickers(ticker, tickersSet):
    BLACKLIST = ['EV','SELL', 'BUY', 'ETF','SPAC','WSB', 'USA', 'IRA']
    
    if ticker in BLACKLIST:
        return False

    ticker = filter_Ticker_Characters(ticker)
    tickersSet.add(ticker)
    return True

for title in titles:
    bools, ticker_word = Mentioned_Potential_Ticker(title)
    file.writerow([title, str(bools), str(ticker_word)])
    print(tickers)
