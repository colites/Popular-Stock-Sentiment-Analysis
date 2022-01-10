import csv
import Stock_Algo.Web_Scrapper_Reddit as Rscrap

## Create the CSV file with UTF-8 Encoding
file = csv.writer(open('Reddit-Titles.csv', 'w',encoding='UTF-8'))
file.writerow(['Title', 'Ticker Mention'])

titles, urls, authors = Rscrap.Get_Submissions_Any(5, 'wallstreetbets')

## Checks whether there is a stock ticker mentioned
## Returns a bool
def Mentioned_Potential_Ticker(title):    
    words = title.split()
    
    ##check each word for $ or for all Uppercase when word is greater than 1
    for word in words:
        if '$' in word:
            return True
        if word.isupper() and len(word) > 1:
            return True
    return False

## Gets the words that were seen to potentially be stock tickers
## Returns a list of tickers
def Extract_Ticker()
    Ticker_Mention = Mentioned_Potential_Ticker(title)

    if Ticker_Mention:
        

#### CAN HAVE MULTIPLE DIFFERENT TICKERS AT ONCE, MUST FIND A WAY TO DIFFERENTIATE
    
for title in titles:
    Ticker_Mention = Mentioned_Potential_Ticker(title)
    file.writerow([title, str(Ticker_Mention)])
