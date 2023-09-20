# Popular-Stock-Sentiment-Analysis

This project is a web application that performs sentiment analysis on popular stock tickers found on Reddit. It scrapes Reddit for mentions of these tickers and collects financial news articles related to them. The application provides various visualizations for sentiment analysis results. The future purpose for this project will be to create a application that can perform sentiment analysis on text along with visual data analytics with a large choice of models in ensemble, all through an easy to use UI that will allow even the less tech savvy people be able to observe analytics for popular stock tickers.

A demo version of this application is deployed on AWS and can be accessed at:

[Link to AWS deployment]

If you want to run the application locally, follow the steps below:

## Prerequisites

- Python 3.7 or newer
- PostgreSQL 15 or newer. You can download it [here](https://www.postgresql.org/download/).

## Installation

1. Clone the repository and navigate to the downloaded folder.
    ```
    git clone https://github.com/yourusername/popular-stock-sentiment-analysis.git
    cd popular-stock-sentiment-analysis
    ```

2. Create a virtual environment and activate it:
    ```
    python3 -m venv env
    source env/bin/activate
    ```

3. Install the required packages, if one command does not work use the other:
    ```
    pip install -r requirements.txt
    py -3 -m pip install -r requirements.txt

    ```

4. Set up PostgreSQL:
    - Install PostgreSQL on your machine following the instructions [here](https://www.postgresql.org/docs/current/install-procedure.html).
    - Create a new PostgreSQL database for the application.
    - Update the `database_config.py` file in your application's configuration with your information. Make sure to keep this information secure.

5. Run the application:
    ```
    py -m flask --app Flask_web run
    ```
    This command will start the server and print an IP address, typically `http://127.0.0.1:5000`, which you can navigate to in your web browser to use the application.

Note: The first time you run the application, it may take some time to download and set up the necessary models. The application that uses a deep learning model (DistilBERT) for sentiment analysis will take much longer if chosen. The first run will include a training step, which may take approximately 55-70 minutes on high-end consumer computers. Subsequent runs will be faster as the trained model will be saved and reused.

## Usage

The application provides a user-friendly interface to interact with. After launching the application, navigate to `http://127.0.0.1:5000` in your web browser to access it. 

On the Input page, you will be prompted to enter the number of Reddit posts to scrape and select the subreddit and the models for analysis. Once you have made your selections, click the "Submit" button to start the analysis.

The application will then scrape the chosen number of Reddit posts, gather the tickers, and then search for financial news related to those tickers. It will then provide sentiment analysis for all the reddit posts and financial articles that mention the tickers. 

You can filter by date by choosing the start and end dates in the date menus. 

You can also view visualizations of the sentiment analysis results. Click on the "Visualizations" button to see various charts representing different information from the database.

Click on the dropdowns to customize the visualization that will be created based on the available data.

The "Mention Information" button will display a table showing the titles of the texts that were evaluated along with the results and the ticker associated with each text.

![Screenshot (54)](https://github.com/colites/Popular-Stock-Sentiment-Analysis/assets/56234676/30473cfb-2dfe-4ae9-ab0a-e8e335019fac)


The "Sentiment Statistics" button will give you an overview of the frequency of each ticker's mention and the associated sentiments for articles and posts dated today.

To perform another analysis, you can always click on the "Back" button to return to the Input page.

Keep in mind that due to the real-time nature of Reddit posts and financial news, the results can change each time you run the analysis.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
