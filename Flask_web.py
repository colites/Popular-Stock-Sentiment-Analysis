from flask import Flask, render_template, request, jsonify
import psycopg2
import psycopg2.extras
from datetime import date
import database_config
import SQL as queries
import visualization as visual 

app = Flask(__name__)
input_values = None


def create_connection_cursor():
    """
    Create a connection and cursor to the database using psycopg2.
    
    Returns:
        tuple: A tuple containing the connection and cursor objects.
    """
    
    connection = psycopg2.connect(host = database_config.DB_HOST,
                                  database = 'hype_stock',
                                  user = database_config.DB_USER,
                                  password = database_config.DB_PASS)
    connection.autocommit = True
    cursor = connection.cursor(cursor_factory = psycopg2.extras.DictCursor)
    
    return connection, cursor


@app.route("/")
def index():
    """
    Render the search input page.

    Returns:
        str: Rendered HTML template for the search input page.
    """
    
    return render_template('Input.html')


@app.route('/loading')
def loading_screen():
    """
    Render the loading screen.

    Returns:
        str: Rendered HTML template for the loading screen.
    """
    
    return render_template('loading.html')

@app.route('/run_process_input', methods=["POST"])
def process_input():
    """
    Receive the user input values from the frontend and store them in a global variable.

    Returns:
        Response: JSON response containing a success message.
    """
    
    global input_values
    
    data = request.get_json()
    num_posts = data.get("num_posts", 1)
    subreddit = data.get("subreddit", "investing")
    models = data.get("models")

    #Javascript and HTML forms convert arrays into strings, so convert it to an array
    models = models.split(',')
    
    input_values = (num_posts, subreddit, models)
    
    return jsonify({"message": "Input values received"})


## Endpoint for the execution of the main script
@app.route('/run')
def create_map():
    """
    Execute the main pipeline script using the input values from the frontend.

    Returns:
        str: 'done' if successful, or a 400 status code with an error message if input values are not received.
    """
     
    import Reddit_filter_and_write as pipe
    global input_values

    if input_values is not None:
        num_posts = input_values[0]
        subreddit = input_values[1]
        models = input_values[2]

        pipe.main_pipeline(num_posts, subreddit, models)
        return f"done"
    else:
        return f"No input values received", 400


@app.route('/mention_information')
def mention_information():
    """
    Retrieve mention information for the current date and return it as a JSON object.

    Returns:
        Response: JSON response containing the mention information.
    """
    
    start_date = request.args.get('start_date', default=None, type=str)
    end_date = request.args.get('end_date', default=None, type=str)
    
    if not start_date or not end_date:
        return jsonify({"error": "Both start_date and end_date parameters are required."}), 400

    connection, cursor = create_connection_cursor()
    data = queries.get_mentions_date_range_query(connection, cursor, start_date, end_date)
    if data == None:
        return jsonify({"error": "No information could be found in this date range"}), 400
    data = [dict(row) for row in data]

    cursor.close()
    connection.close()
    return jsonify(data)


@app.route('/sentiment_statistics')
def sentiment_statistics():
    """
    Retrieve sentiment statistics for the current date and return them as a JSON object.

    Returns:
        Response: JSON response containing the sentiment statistics.
    """
    
    start_date = request.args.get('start_date', default=None, type=str)
    end_date = request.args.get('end_date', default=None, type=str)
    
    if not start_date or not end_date:
        return jsonify({"error": "Both start_date and end_date parameters are required."}), 400

    connection, cursor = create_connection_cursor()
    data = queries.compare_sentiments_all_stocks_date_range(connection, cursor, start_date, end_date)
    if data == None:
        return jsonify({"error": "No information could be found in this date range"}), 400
    data = [dict(row) for row in data]
   
    cursor.close()
    connection.close()
    return jsonify(data)


@app.route('/output')
def button_click():
    """
    Render the results page.

    Returns:
        str: Rendered HTML template for the results page.
    """
    
    return render_template('output.html')


@app.route('/symbols')
def get_symbols():
    """
    Retrieve all stock symbols from the database and return them as a JSON object.

    Returns:
        Response: JSON response containing the stock symbols.
    """

    connection, cursor = create_connection_cursor()
    symbols = queries.find_all_stock_symbols_alltime(connection, cursor)
    return jsonify(symbols)


@app.route('/get_bar_graph')
def get_bar_graphs():
    """
    Retrieve bar graph data for a specific stock symbol and return it as a JSON object.

    Returns:
        Response: JSON response containing the bar graph data.
    """

    # Get the y-axis parameter from the request URL
    measured_y = request.args.get('y_measure_type', default=None, type=str)
    if not measured_y:
            return jsonify({"error": "Variable to measure not specified"}), 400

    # Get the x-axis parameter if needed. 
    if measured_y not in ["Sentiments"]:
        measured_x = request.args.get('x_measure_type', default=None, type=str)
        if not measured_x:
            return jsonify({"error": "Variable to be measured not specified"}), 400
        
    # Get the date query parameters from the request URL
    start_date = request.args.get('start_date', default=None, type=str)
    end_date = request.args.get('end_date', default=None, type=str)
    if not start_date or not end_date:
        return jsonify({"error": "Both start_date and end_date parameters are required."}), 400

    # Get the source_type query parameter from the request URL
    source_type = request.args.get('source_type', default=None, type=str)
    if not source_type:
        return jsonify({"error": "Source type not specified"}), 400
    
    # Get the symbol query parameter from the request URL
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol not specified"}), 400

    
    connection, cursor = create_connection_cursor()
    data = queries.compare_sentiments_stock_date_range(connection,
                                                       cursor,
                                                       symbol,
                                                       start_date,
                                                       end_date,
                                                       measured_y,
                                                       source_type)

    cursor.close()
    connection.close()
    print(data)

    if data is None:
        return jsonify({"error": "No data available"})
    
    mentions = [data[0]]
    amounts = [data[4]] 
    psentiments = [data[1]] 
    nsentiments = [data[2]]
    neutral_sentiments= [data[3]]

    encoded_image = visual.mentions_and_sentiments_barGraph(mentions, amounts, psentiments, nsentiments, neutral_sentiments)

    return jsonify({"image": encoded_image})


@app.route('/get_pie_chart')
def get_pie_charts():
    """
    Retrieve pie chart data for stock mentions and return it as a JSON object.

    Returns:
        Response: JSON response containing the pie chart data.
    """

    # Get the y-axis parameter from the request URL
    measured_y = request.args.get('y_measure_type', default=None, type=str)
    if not measured_y:
            return jsonify({"error": "Variable to measure not specified"}), 400

    # Get the x-axis parameter if needed. 
    if measured_y not in ["Positive", "Negative", "Neutral", "Total"]:
        measured_x = request.args.get('x_measure_type', default=None, type=str)
        if not measured_x:
            return jsonify({"error": "Variable to be measured not specified"}), 400
        
    # Get the date query parameters from the request URL
    start_date = request.args.get('start_date', default=None, type=str)
    end_date = request.args.get('end_date', default=None, type=str)
    if not start_date or not end_date:
        return jsonify({"error": "Both start_date and end_date parameters are required."}), 400

    # Get the source_type query parameter from the request URL
    source_type = request.args.get('source_type', default=None, type=str)
    if not source_type:
        return jsonify({"error": "Source type not specified"}), 400

    # Get the symbol query parameter from the request URL
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol not specified"}), 400
    
    connection, cursor = create_connection_cursor()
    data = queries.total_mentions_date_range_query(connection, cursor, start_date, end_date)

    cursor.close()
    connection.close()

    symbol_data = [row['symbol'] for row in data]
    if measured_y == "Positive":
        amounts = [row['positive_mentions'] for row in data]
    elif measured_y == "Negative":
        amounts = [row['negative_mentions'] for row in data]
    elif measured_y == "Neutral":
        amounts = [row['neutral_mentions'] for row in data]
    else:
        amounts = [row['total_mentions'] for row in data]

    encoded_image = visual.piechart_mentions(symbol_data, amounts)

    return jsonify({"image": encoded_image})


@app.route('/get_donut_chart')
def get_donut_charts():
    """
    Retrieve donut chart data for stock mentions and return it as a JSON object.

    Returns:
        Response: JSON response containing the donut chart data.
    """

    # Get the y-axis parameter from the request URL
    measured_y = request.args.get('y_measure_type', default=None, type=str)
    if not measured_y:
            return jsonify({"error": "Variable to measure not specified"}), 400

    # Get the x-axis parameter if needed. 
    if measured_y not in ["Positive", "Negative", "Neutral", "Total"]:
        measured_x = request.args.get('x_measure_type', default=None, type=str)
        if not measured_x:
            return jsonify({"error": "Variable to be measured not specified"}), 400
        
    # Get the date query parameters from the request URL
    start_date = request.args.get('start_date', default=None, type=str)
    end_date = request.args.get('end_date', default=None, type=str)
    if not start_date or not end_date:
        return jsonify({"error": "Both start_date and end_date parameters are required."}), 400

    # Get the source_type query parameter from the request URL
    source_type = request.args.get('source_type', default=None, type=str)
    if not source_type:
        return jsonify({"error": "Source type not specified"}), 400
    
    # Get the symbol query parameter from the request URL
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol not specified"}), 400
    
    connection, cursor = create_connection_cursor()
    data = queries.total_mentions_date_range_query(connection, cursor, start_date, end_date)

    cursor.close()
    connection.close()

    symbol_data = [row['symbol'] for row in data]
    if measured_y == "Positive":
        amounts = [row['positive_mentions'] for row in data]
    elif measured_y == "Negative":
        amounts = [row['negative_mentions'] for row in data]
    elif measured_y == "Neutral":
        amounts = [row['neutral_mentions'] for row in data]
    else:
        amounts = [row['total_mentions'] for row in data]

    encoded_image = visual.donutchart_mentions(symbol_data, amounts)

    return jsonify({"image": encoded_image})


@app.route('/get_line_graph')
def get_line_graphs():
    """
    Retrieve line graph data for a specific stock symbol and return it as a JSON object.

    Returns:
        Response: JSON response containing the bar graph data.
    """

    # Get the y-axis parameter from the request URL
    measured_y = request.args.get('y_measure_type', default=None, type=str)
    if not measured_y:
            return jsonify({"error": "Variable to measure not specified"}), 400

    # Get the x-axis parameter if needed. 
    if measured_y not in ["Sentiments"]:
        measured_x = request.args.get('x_measure_type', default=None, type=str)
        if not measured_x:
            return jsonify({"error": "Variable to be measured not specified"}), 400
        
    # Get the date query parameters from the request URL
    start_date = request.args.get('start_date', default=None, type=str)
    end_date = request.args.get('end_date', default=None, type=str)
    if not start_date or not end_date:
        return jsonify({"error": "Both start_date and end_date parameters are required."}), 400

    # Get the source_type query parameter from the request URL
    source_type = request.args.get('source_type', default=None, type=str)
    if not source_type:
        return jsonify({"error": "Source type not specified"}), 400
    
    # Get the symbol query parameter from the request URL
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol not specified"}), 400
    
    connection, cursor = create_connection_cursor()
    data = queries.compare_sentiments_stock_date_range(connection,
                                                       cursor,
                                                       symbol,
                                                       start_date,
                                                       end_date,
                                                       measured_y,
                                                       source_type)

    cursor.close()
    connection.close()

    if data is None:
        return jsonify({"error": "No data available"})
    
    mentions = [data[0]]
    amounts = [data[4]] 
    psentiments = [data[1]] 
    nsentiments = [data[2]]
    neutral_sentiments= [data[3]]

    encoded_image = visual.mentions_and_sentiments_line_graph(mentions, amounts, psentiments, nsentiments, neutral_sentiments)

    return jsonify({"image": encoded_image})


if __name__ == '__main__':
    app.run(debug=False)
