Steps to begin using this code 

1. run requirements.txt in its directory to install all dependencies
- Command line: pip install -r requirements.txt  
- if Python is not in path:py -3 -m pip install -r requirements.txt
- preferably run these requirements in a environment to avoid version conflicts.

2. Download the data to train the model
- Command line: py -m nltk.downloader movie_reviews
- interactive or manual installation of data at https://www.nltk.org/data.html

3. run webcrawl_product.py
- Command line: py Reddit_filter_and_write.py