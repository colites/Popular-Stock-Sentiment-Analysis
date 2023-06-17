# Popular-Stock-Sentiment-Analysis

This is a web application that provides multiple visualization techniques for sentiment analysis. on scrapes Reddit for popular stock tickers and then looks for financial news articles related to those tickers.  

## Installation

A demo version is deployed in AWS with the link: 

                                             aslngaslkdngklasndglknaslg

Otherwise, this application can be installed by downloading the code and preferably running the application in an environment to avoid version conflicts.

To install and use the application, follow these steps:

Install PostgreSQL to be able to store data. Installation instructions can be found on the Postgres SQl site https://www.postgresql.org/docs/current/install-procedure.html

run requirements.txt in its directory to install all dependencies. 
If one command does not work, try the other command
```bash
pip install -r requirements.txt
py -3 -m pip install -r requirements.txt
```

Once the required packages from the text file are installed, you now have to run the server using flask. 

```bash
py -m flask --app Flask_web run
```

You will be given a IP corresponding to where the flask app is running from, which would be the client machine (your machine) in this case. Navigate to the IP in your browser, the default IP usually being http://127.0.0.1:5000

Running the application for the first time will be lengthy due to the downloading of files depending on which models are used. If the deep learning model Distilbert is run, then training times in high-end consumer computers will be ~30-45 minutes and can be longer for the first run. It will be much shorter in subsequent runs.
