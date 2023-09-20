import io
import base64
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def calculate_font_size(num_vars):
    """
    Calculate the font size based on the number of variables (mentions)
    
    The font size is reduced as the number of variables increases to prevent label overlap.
    
    Args:
        num_vars (int): The number of variables (mentions) in the chart.
    
    Returns:
        int: The calculated font size.
    """

    base_size = 13
    if num_vars <= 10:
        return base_size
    return max(base_size - 0.5 *(num_vars - 10), 6)


def generate_labels(bars,ax):
    """
    Add data labels above each bar in a bar graph.
    
    Args:
        bars (iterable): An iterable containing the bars in the bar graph.
        ax: The axes object to which the bars belong.
    """
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


def image_base_encoded(fig):
    """
    Convert a matplotlib figure to a base64-encoded image.
    
    Args:
        fig: The figure object to convert.
    
    Returns:
        str: The base64-encoded image.
    """
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return encoded_image


def mentions_and_sentiments_barGraph(mentions, amounts, psentiments, nsentiments, neutral_sentiments):
    """
    Create a bar graph comparing the total, positive, negative, and neutral mentions of a specific stock.
    
    Args:
        mentions (list): A list of stock tickers.
        amounts (list): A list of total mentions for each stock ticker.
        psentiments (list): A list of positive sentiment counts for each stock ticker.
        nsentiments (list): A list of negative sentiment counts for each stock ticker.
        neutral_sentiments (list): A list of neutral sentiment counts for each stock ticker.
    
    Returns:
        str: The base64-encoded image of the bar graph.
    """
    
    fig, ax = plt.subplots(figsize=(8,8))

    positive_mentions = ax.bar(0.5, psentiments, 0.25, label ='Positive', color = 'green')
    negative_mentions = ax.bar(0.75, nsentiments, 0.25, label ='Negative', color = 'red')
    neutral_mentions = ax.bar(0.25, neutral_sentiments, 0.25, label ='Neutral', color = 'gray')
    total_mentions = ax.bar(1, amounts, 0.25, label =' Total', color = 'black')

    ax.set_xlabel('Stock Ticker')
    ax.set_ylabel('Number of mentions')
    ax.set_title('Stock ticker sentiment totals')
    ax.set_xticks([1])
    ax.set_xticklabels(mentions)
    ax.legend()
    
    ax.set_xlim(-0.5, 2.5)
    
    generate_labels(total_mentions,ax)
    generate_labels(positive_mentions,ax)
    generate_labels(negative_mentions,ax)
    generate_labels(neutral_mentions,ax)

    return image_base_encoded(fig)


                ########## NOT DONE ###########

def stocks_sentiments_barGraph_all(mentions, amounts, psentiments, nsentiments, neutral_sentiments):
    """
    Create a bar graph comparing positive and negative sentiment counts for multiple stock tickers.

    Args:
        mentions (list): A list of stock tickers.
        amounts (list): A list of total mentions for each stock ticker.
        psentiments (list): A list of positive sentiment counts for each stock ticker.
        nsentiments (list): A list of negative sentiment counts for each stock ticker.
        neutral_sentiments (list): A list of neutral sentiment counts for each stock ticker.
    
    Returns:
        str: The base64-encoded image of the bar graph.
    """
    
    x = np.arange(10)
    
    fig, ax = plt.subplots(figsize=(8,8))

    positive_mentions = ax.bar(x - 0.1, psentiments, 0.25, label ='Positive', color = 'green')
    negative_mentions = ax.bar(x + 0.1, nsentiments, 0.25, label ='Negative', color = 'red')
    total_mentions = ax.bar(x, amounts, 0.15, label =' Total', color = 'black')

    ax.set_xlabel('Stock Ticker')
    ax.set_ylabel('Number of mentions')
    ax.set_title('Comparison of Positive and Negative Mentions in stock tickers')
    ax.set_xticks(mentions)
    ax.set_xticklabels(mentions)
    ax.legend()
    
    generate_labels(total_mentions,ax)
    generate_labels(positive_mentions,ax)
    generate_labels(negative_mentions,ax)
    generate_labels(neutral_mentions,ax)

    return image_base_encoded(fig)

            ################################################
    
def piechart_mentions(mentions, amounts):
    """
    Create a pie chart showing the proportion of total mentions for each stock ticker.
    
    Args:
        mentions (list): A list of stock tickers.
        amounts (list): A list of total mentions for each stock ticker.
    
    Returns:
        str: The base64-encoded image of the pie chart.
    """
    
    font_size = calculate_font_size(len(mentions))

    fig, ax = plt.subplots(figsize=(8,8))   
    ax.pie(amounts, labels= mentions, shadow = True, startangle = 90, autopct='%1.1f%%',
           textprops={'fontsize': font_size})

    ax.set_title("Total Mentions for each stock")
    ax.axis('equal')
    return image_base_encoded(fig)


def donutchart_mentions(mentions, amounts):
    """
    Create a donut chart showing the proportion of total mentions for each stock ticker.
    
    Args:
        mentions (list): A list of stock tickers.
        amounts (list): A list of total mentions for each stock ticker.
    
    Returns:
        str: The base64-encoded image of the donut chart.
    """
    
    font_size = calculate_font_size(len(mentions))

    fig, ax = plt.subplots(figsize=(8,8))
    ax.pie(amounts, labels= mentions, startangle = 90, autopct='%1.1f%%',
           textprops={'fontsize': font_size})
    
    centre_circle = plt.Circle((0,0),0.90,fc='white')
    fig.gca().add_artist(centre_circle)

    ax.set_title("Total Mentions for each stock")
    ax.axis('equal')

    return image_base_encoded(fig)


def piechart_sentiments(sentiments, values):
    """
    Create a pie chart showing the proportion of sentiments for a stock.
    
    Args:
        sentiments (list): A list of sentiment types.
        values (list): A list of counts for each sentiment type.
    
    Returns:
        str: The base64-encoded image of the pie chart.
    """
    
    font_size = calculate_font_size(len(values))
    colors = ['#008000', '#FF0000', '#808080']  # Green for positive, Red for negative, Grey for neutral

    fig, ax = plt.subplots(figsize=(8,8))   
    # Set labeldistance to 0.7 to prevent label overlap
    wedges, texts, autotexts = ax.pie(values, labels=sentiments, colors=colors, shadow=True, startangle=90, autopct='%1.1f%%', labeldistance=0.7)

    ax.set_title("Sentiment Proportions for the Stock")
    ax.axis('equal')

    legend_labels = ['Positive', 'Negative', 'Neutral']
    ax.legend(legend_labels, loc='upper right', fontsize=font_size)

    return image_base_encoded(fig)


def donutchart_sentiments(sentiments, values):
    """
    Create a donut chart showing the proportion of sentiments for a stock.
    
    Args:
        sentiments (list): A list of sentiment types.
        values (list): A list of counts for each sentiment type.
    
    Returns:
        str: The base64-encoded image of the donut chart.
    """
    
    font_size = calculate_font_size(len(values))
    colors = ['#008000', '#FF0000', '#808080']  # Green for positive, Red for negative, Grey for neutral

    fig, ax = plt.subplots(figsize=(8,8))

    # Set labeldistance to 0.7 to prevent label overlap
    wedges, texts, autotexts = ax.pie(values, labels=sentiments, colors=colors, shadow=True, startangle=90, autopct='%1.1f%%', labeldistance=0.7)

    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)

    ax.set_title("Sentiment Proportions for the Stock")
    ax.axis('equal')

    legend_labels = ['Positive', 'Negative', 'Neutral']
    ax.legend(legend_labels, loc='upper right', fontsize=font_size)

    # Adjust font size for text and autopct
    for text in texts:
        text.set(size=font_size)
    for autotext in autotexts:
        autotext.set(size=font_size)
        
    return image_base_encoded(fig)


def mentions_and_sentiments_line_graph(mentions, amounts, psentiments, nsentiments, neutral_sentiments):
    """
    Create a line graph comparing the total, positive, negative, and neutral mentions of a specific stock.
    
    Args:
        mentions (list): A list of stock tickers.
        amounts (list): A list of total mentions for each stock ticker.
        psentiments (list): A list of positive sentiment counts for each stock ticker.
        nsentiments (list): A list of negative sentiment counts for each stock ticker.
        neutral_sentiments (list): A list of neutral sentiment counts for each stock ticker.
    
    Returns:
        str: The base64-encoded image of the line graph.
    """
    
    fig, ax = plt.subplots(figsize=(8,8))
    
    ax.plot(mentions, psentiments, color='green', marker='o', label='Positive')
    ax.plot(mentions, nsentiments, color='red', marker='o', label='Negative')
    ax.plot(mentions, neutral_sentiments, color='gray', marker='o', label='Neutral')
    ax.plot(mentions, amounts, color='black', marker='o', label='Total')
    
    ax.set_xlabel('Stock Ticker')
    ax.set_ylabel('Number of mentions')
    ax.set_title('Stock ticker sentiment totals')
    ax.legend()
    
    # annotate points on the line graph
    for i, txt in enumerate(psentiments):
        ax.annotate(txt, (mentions[i], psentiments[i]), fontsize=9, ha='center', va='bottom')
        
    for i, txt in enumerate(nsentiments):
        ax.annotate(txt, (mentions[i], nsentiments[i]), fontsize=9, ha='center', va='bottom')
        
    for i, txt in enumerate(neutral_sentiments):
        ax.annotate(txt, (mentions[i], neutral_sentiments[i]), fontsize=9, ha='center', va='bottom')
        
    for i, txt in enumerate(amounts):
        ax.annotate(txt, (mentions[i], amounts[i]), fontsize=9, ha='center', va='bottom')
    
    return image_base_encoded(fig)
