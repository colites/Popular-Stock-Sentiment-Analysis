# Popular-Stock-Sentiment-Analysis

This project is a web application that performs sentiment analysis on popular stock tickers found on Reddit. It scrapes Reddit for mentions of these tickers and collects financial news articles related to them. The application provides various visualizations for sentiment analysis results.

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

Note: The first time you run the application, it may take some time to download and set up the necessary models. The application that uses a deep learning model (DistilBERT) for sentiment analysis will take much longer if chosen. The first run will include a training step, which may take approximately 30-45 minutes on high-end consumer computers. Subsequent runs will be faster as the trained model will be saved and reused.

## Usage

- Explain how to use the application here
