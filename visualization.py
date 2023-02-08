import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## read the csv data 
data = pd.read_csv('Reddit-Titles-Stats.csv')
data_mentions_tickers = data['ticker']
data_mentions_amounts = data['total_mentions']
data_mentions_positive_sentiments = data['psentiment']
data_mentions_negative_sentiments = data['nsentiment']

##generates the label above the bars in the bargraphs
def generate_labels(bars,ax):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points")

## makes a visualization in the form of a bar graph
## comparison between positive and negative sentiments found for one stock
def mentions_and_sentiments_barGraph(mentions, amounts, psentiments, nsentiments):
    nmentions = mentions[0:10]
    namounts = amounts[0:10]
    psentiments = psentiments[0:10]
    nsentiments = nsentiments[0:10]
    x = np.arange(10)
    
    fig, ax = plt.subplots()
    positive_mentions = ax.bar(x - 0.1, psentiments, 0.25, label ='Positive', color = 'green')
    negative_mentions = ax.bar(x + 0.1, nsentiments, 0.25, label ='Negative', color = 'red')
    total_mentions = ax.bar(x, namounts, 0.15, label =' Total', color = 'black')
    ax.set_xlabel('Stock Ticker')
    ax.set_ylabel('Number of mentions')
    ax.set_title('Comparison of Positive and Negative Mentions in stock tickers')
    ax.set_xticks(x)
    ax.set_xticklabels(nmentions)
    ax.legend()
    
    generate_labels(total_mentions,ax)
    plt.show()

## makes a visualization in the form of a pie chart 
## shows total mentions for each stock ticker in one search
def piechart_mentions(mentions, amounts):
    nmentions = mentions[0:10]
    nmentions = nmentions.reset_index(drop = True)
    namounts = amounts[0:10]
    namounts = namounts.reset_index(drop = True)
    
    fig2, ax2 = plt.subplots()
    ax2.pie(namounts, labels= nmentions, shadow = True, startangle = 90)
    ax2.axis('equal')
    plt.show()

## makes a visualization in the form of a donut chart 
## shows total mentions for each stock ticker in one search
def donutchart_mentions(mentions, amounts):
    nmentions = mentions[0:10]
    nmentions = nmentions.reset_index(drop = True)
    namounts = amounts[0:10]
    namounts = namounts.reset_index(drop = True)

    fig2, ax2 = plt.subplots()
    ax2.pie(namounts, labels= nmentions, startangle = 90,autopct='%1.1f%%')
    
    centre_circle = plt.Circle((0,0),0.90,fc='white')
    fig2.gca().add_artist(centre_circle)
    
    ax2.axis('equal')
    plt.show()

piechart_mentions(data_mentions_tickers, data_mentions_amounts)
donutchart_mentions(data_mentions_tickers, data_mentions_amounts)
mentions_and_sentiments_barGraph(data_mentions_tickers, 
								data_mentions_amounts,
								data_mentions_positive_sentiments,
								data_mentions_negative_sentiments)
