import requests
from datetime import date, datetime
import yfinance as yf
from bs4 import BeautifulSoup


def Get_Submissions_Any(size, subreddit):
    """
    Fetch the indicated amount of submissions from the specified subreddit using Reddit's top endpoint.
    
    Args:
        size (int): The number of submissions to fetch.
        subreddit (str): The subreddit to fetch submissions from.
    
    Returns:
        list: A list of tuples, each containing the title, url, and author of a submission.
    """
    
    submission_list = []
    url = "https://www.reddit.com/r/" + subreddit + "/new/.json?limit=" + str(size)

    response = requests.get(url, headers={"User-Agent": "wgah/5.0"}, timeout=20)

    if response.status_code == 200:
        data = response.json()["data"]["children"]

        for post in data:
            title = post["data"]["title"]
            url = post["data"]["url"]
            author = post["data"]["author"]
            date = datetime.fromtimestamp(post["data"]["created_utc"])
            
            submission_list.append((title,url,author,date))
    else:
        print(f"Error fetching data from Reddit API: {response.status_code}")

    return submission_list

    
def Get_stock_data(stock_ticker):
    """
    Fetch stock data for the given stock ticker using Yahoo Finance.
    
    Args:
        stock_ticker (str): The stock ticker to fetch data for.
    
    Returns:
        tuple: A tuple containing the current date, the day's low price, and the day's high price.
               Returns (None, None, None) if the stock ticker is not found.
    """
    
    stock_ticker_clean = stock_ticker.replace("$","")
    
    try:
        ticker = yf.Ticker(stock_ticker_clean)
        if ticker:
            return date.today(), ticker.fast_info['day_low'], ticker.fast_info['day_high']
    except Exception as e:
        print(f"Error fetching data from Yahoo Finance: {e}")

    return None, None, None


def scrape_yahoo_finance(ticker):
    """
    Scrape financial news related to a specific stock ticker from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        list: A list of tuples containing the title, link, author, and date of each news article.
    """
    
    url = f'https://finance.yahoo.com/quote/{ticker}/news?p={ticker}'

    # Yahoo finance needs a header so that it does not 404
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}

    response = requests.get(url, headers=headers, timeout=5)
    if not response.ok:
        print('Status code:', response.status_code)

    soup = BeautifulSoup(response.text, 'html.parser')

    result = []

    articles = soup.find_all('li', {'class': 'js-stream-content'})
    
    # The text title is not optional, everything else is
    for article in articles:
        try:
            title = article.find('h3').text.strip()
            
            link = article.find('a')
            link = link['href'] if link is not None else 'unknown'
            
            author = article.find('div', {'class': 'C(#959595)'})
            author = author.text if author is not None else 'unknown'

            # The date is dynamically generated relative to viewers time
            # I do not want to use something like selenium just to get the date, the next best thing is todays date
            article_date = date.today()
            
            result.append((title, link, author, article_date))
        except AttributeError:
            continue
        
    return result


def scrape_marketwatch(ticker):
    """
    Scrape financial news related to a specific stock ticker from MarketWatch.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        list: A list of tuples containing the title, link, author, and date of each news article.
    """
    
    url = f'https://www.marketwatch.com/investing/stock/{ticker}/news'
    response = requests.get(url, timeout=5)
    if not response.ok:
        print('Status code:', response.status_code)
        
    soup = BeautifulSoup(response.text, 'html.parser')

    result = []

    articles = soup.find_all('div', {'class': 'article__content'})

    # The text title is not optional, everything else is
    for article in articles:
        try:
            title = article.find('h3', {'class': 'article__headline'}).text.strip()

            link = article.find('a')
            link = link['href'] if link is not None else 'unknown'
            if link == '#':
                continue
            
            author = article.find('p', {'class': 'author__name'})
            author = author.text.strip() if author is not None else 'unknown'

            # Get the date and convert it to a datetime object
            date_string = article.find('span', class_='article__timestamp').text.strip()  
            date_string = date_string.split(' at ')[0]
            date_string = date_string.replace('.', '')
            date = datetime.strptime(date_string, '%b %d, %Y')
            
            result.append((title, link, author, date))
        except AttributeError:
            continue
        
    return result


def get_financial_news(ticker):
    """
    Get financial news related to a specific stock ticker from both Yahoo Finance and MarketWatch.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        list: A list of tuples containing the title, link, and author of each news article from both websites.
    """
    
    yahoo_results = scrape_yahoo_finance(ticker)
    marketwatch_results = scrape_marketwatch(ticker)
    
    return yahoo_results + marketwatch_results
