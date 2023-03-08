import time
from flask import Flask
from flask import render_template
from flask import request
import Reddit_filter_and_write as pipe

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('Input.html')

@app.route('/loading')
def loading_screen():
    return render_template('loading.html')

@app.route('/run')
def create_map():
    pipe.main(60)
    return "done"

@app.route('/output')
def button_click():
    return render_template('output.html')

if __name__ == '__main__':
    app.run(debug=False)
