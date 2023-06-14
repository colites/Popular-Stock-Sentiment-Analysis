Steps to begin using this code 

1. run requirements.txt in its directory to install all dependencies
- Command line: pip install -r requirements.txt  
- if Python is not in path:py -3 -m pip install -r requirements.txt
- preferably run these requirements in a environment to avoid version conflicts.

2. run webcrawl_product.py
- Command line: py Reddit_filter_and_write.py

3. After setup, running the application for the first time will be lengthy due to the downloading of some files
from the deep learning package.

-- new:
- type in the command: py -m flask --app Flask_web run