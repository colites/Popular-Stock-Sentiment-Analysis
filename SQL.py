import psycopg2
import database_config

def create_tables(connection, cursor):
    """
    Create tables in the PostgreSQL database if they do not exist
    and create an index on the table with the dates for faster querying.

    Args:
        connection: A connection object representing the database connection
        cursor: A cursor object to execute database queries
    
    """
    
    tables = (
        """
        CREATE TABLE IF NOT EXISTS stock_info (
            entry_id SERIAL PRIMARY KEY,
            symbol VARCHAR(6) NOT NULL UNIQUE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS mentions (
                mention_id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                symbol VARCHAR(6) NOT NULL,
                mention_text TEXT NOT NULL,
                low NUMERIC NOT NULL,
                high NUMERIC NOT NULL,
                source TEXT NOT NULL,
                source_type TEXT NOT NULL,
                UNIQUE (date, mention_text),
                FOREIGN KEY (symbol) REFERENCES stock_info (symbol) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS stock_sentiments (
                sentiments_id SERIAL PRIMARY KEY,
                sentiment VARCHAR(8) NOT NULL,
                FOREIGN KEY (sentiments_id) REFERENCES mentions (mention_id) ON DELETE CASCADE
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_mentions_date ON mentions (date)
        """)
    try:
        for table in tables:
            cursor.execute(table)
            
        print(f"Tables created/validated successfully")
    except Exception as err:
        print(f"failed to create tables")
        print(f"Error: '{err}'")

        
def create_database(connection, cursor):
    """
    Create a PostgreSQL database if it does not exist and connect to it.

    Args:
        connection: A connection object representing the database connection
        cursor: A cursor object to execute database queries

    Returns:
        connection: A new connection object
        cursor: A new cursor object
    """
    
    try:
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'hype_stock'")
        exists = cursor.fetchone()
        if not exists:
            cursor.execute("CREATE DATABASE hype_stock WITH ENCODING 'UTF8'")
            print("Database created successfully")

        else:
            print("Database exists already")

        ## close the connection to connect to the right database
        cursor.close()
        connection.close()

        connection = psycopg2.connect(host = database_config.DB_HOST, database = 'hype_stock', user = database_config.DB_USER, password = database_config.DB_PASS)
        connection.autocommit = True
        cursor = connection.cursor(cursor_factory = psycopg2.extras.DictCursor)
        
        return connection, cursor
    except Exception as err:
        print("failed to create database")
        print(f"Error: '{err}'")


def insert_stock(connection, cursor, stock_data):
    """
    inserts a stock into the stock_info table in the database.

    Args:
        connection: A connection object representing the database connection
        cursor: A cursor object to execute database queries
        stock_data (tuple): A tuple containing (date, symbol, low, high) stock information.
        
    """
    
    try:
        symbol = stock_data[1]  
        cursor.execute("""
            INSERT INTO stock_info (symbol)
            VALUES (%s) ON CONFLICT (symbol) DO NOTHING""", (symbol,))
        
        print("Stock search info inserted into database")
    except Exception as err:
        print("failed to insert stock into database")
        print(f"Error: '{err}'")


###For now, use autocommit but autocommit might end up creating errors, so change later.
def insert_mention_and_sentiment(connection, cursor, stock_data, sentiment, text, date, source_type):
    """
    Insert a mention and sentiment into their respective tables in the database.
    These tables are one-to-one related, so if one fails, both have to be stopped.

    Args:
        connection: A connection object representing the database connection.
        cursor: A cursor object to execute database queries.
        stock_data (tuple): A tuple containing (date, symbol, low, high) stock information.
        sentiment (str): A string representing the sentiment ('positive', 'negative', or 'neutral').
        text (str): A string representing the mention text.
    
    """
    
    try:
        date_checked, symbol, low, high = stock_data[0], stock_data[1], stock_data[2], stock_data[3]   
        cursor.execute("""
            INSERT INTO mentions (date, symbol, mention_text, low, high, source, source_type)
            VALUES (%s, %s, %s, %s, %s, 'HiddenForPrivacy', %s) RETURNING mention_id""",
            (date, symbol, text, low, high, source_type))
        
        mention_id = cursor.fetchone()[0]

        cursor.execute("""
            INSERT INTO stock_sentiments (sentiments_id, sentiment)
            VALUES (%s, %s)""",
            (mention_id, sentiment))
        
        print("Stock mention and sentiment info inserted into the database")
    except Exception as err:
        print("failed to insert stock mention or sentiment into the database")
        print(f"Error: '{err}'")


def find_all_stock_symbols_alltime(connection, cursor):
    """
    Query to find all symbols in the stock_info table.

    Args:
        connection: A connection object representing the database connection
        cursor: A cursor object to execute database queries

    Returns:
        list: A list of symbols in stock_info
    """
    
    query = """
        SELECT symbol
        FROM stock_info
        ORDER BY symbol ASC;
    """
    cursor.execute(query)
    result = cursor.fetchall()
    return result


def find_all_stock_symbols_date_range(connection, cursor, start_date, end_date):
    """
    Query to find all stock symbols mentioned within a specific date range.

    Args:
        connection (object): A connection object representing the database connection.
        cursor (object): A cursor object used to execute database queries.
        start_date (str): The start date of the range for which symbols are queried.
        end_date (str): The end date of the range for which symbols are queried.

    Returns:
        list: A list of symbols mentioned within the specified date range.
    """
    
    query = """
        SELECT m.symbol
        FROM mentions m
        WHERE m.date >= %s AND m.date <= %s
        ORDER BY m.symbol ASC;
    """
    cursor.execute(query, (start_date, end_date))
    result = cursor.fetchall()
    return result


def total_mentions_date_range_query(connection, cursor, start_date, end_date):
    """
    Query for the total number of mentions of each stock for the current date.

    Args:
        connection: A connection object representing the database connection
        cursor: A cursor object to execute database queries
        date: A date object representing the current date
    
    Returns:
        list: A list of dictionaries containing the stock symbol and total_mentions
    """
    
    query = """
        SELECT
            m.symbol,
            SUM(CASE WHEN ss.sentiment = 'positive' THEN 1 ELSE 0 END) AS positive_mentions,
            SUM(CASE WHEN ss.sentiment = 'negative' THEN 1 ELSE 0 END) AS negative_mentions,
            SUM(CASE WHEN ss.sentiment = 'neutral' THEN 1 ELSE 0 END) AS neutral_mentions,
            COUNT(m.mention_id) as total_mentions
        FROM mentions m
        WHERE m.date >= %s AND m.date <= %s
        GROUP BY m.symbol;
    """
    cursor.execute(query, (start_date, end_date))
    result = cursor.fetchall()
    return result


def mentions_over_time_stock_query(connection, cursor):
    """
    Query for the number of mentions for each stock over time.

    Args:
        connection: A connection object representing the database connection
        cursor: A cursor object to execute database queries

    Returns:
        list: A list of dictionaries containing the date, stock symbol, and mentions_count
    """
    
    query = """
        SELECT m.date, m.symbol, COUNT(m.mention_id) as mentions_count
        FROM mentions m
        GROUP BY m.date, m.symbol
        ORDER BY m.date, m.symbol;
    """
    cursor.execute(query)
    result = cursor.fetchall()
    return result


def compare_sentiments_stock_query(connection, cursor, symbol):
    """
    Query for the positive, negative, and neutral sentiments for a given stock.
    Also gets the total amount of mentions for the stock.

    Args:
        connection: A connection object representing the database connection
        cursor: A cursor object to execute database queries
        symbol (str): A string representing the stock symbol.

    Returns:
        dict: A dictionary containing the stock symbol, positive_mentions, negative_mentions, neutral_mentions, and total_mentions
    """
    
    query = """
        SELECT
            m.symbol,
            SUM(CASE WHEN ss.sentiment = 'positive' THEN 1 ELSE 0 END) AS positive_mentions,
            SUM(CASE WHEN ss.sentiment = 'negative' THEN 1 ELSE 0 END) AS negative_mentions,
            SUM(CASE WHEN ss.sentiment = 'neutral' THEN 1 ELSE 0 END) AS neutral_mentions,
            COUNT(ss.sentiments_id) AS total_mentions
        FROM mentions m
        JOIN stock_sentiments ss ON m.mention_id = ss.sentiments_id
        WHERE m.symbol = %s
        GROUP BY m.symbol;
    """
    cursor.execute(query, (symbol,))
    result = cursor.fetchone()
    return result


########## REMEMBER TO RECHECK THE DOCS, LOTS OF THEM ARE OFF Now
def get_mentions_date_range_query(connection, cursor, start_date, end_date):
    """
    Query to see the mentions that have been classified within a date range along with their classification

    Args:
        connection: A connection object representing the database connection
        cursor: A cursor object to execute database queries
        start_date: A date object representing the start of the date range
        end_date: A date object representing the end of the date range

    Returns:
        list: A list of dictionaries containing the stock symbol, mention_text, and sentiment
    """
    
    query = """
        SELECT
            m.date,
            m.symbol,
            m.mention_text,
            ss.sentiment,
            m.source_type
        FROM mentions m
        JOIN stock_sentiments ss ON m.mention_id = ss.sentiments_id
        WHERE m.date >= %s AND m.date <= %s
        ORDER BY m.date;
    """
    cursor.execute(query, (start_date, end_date))
    result = cursor.fetchall()
    return result

### change docstrings
def compare_sentiments_stock_date_range(connection, cursor, symbol, start_date, end_date, measured_variable, source_type='All'):
    """
    Query for the positive, negative, and neutral sentiments for a given stock on the given date range
    Also gets the total amount of mentions for the stock.

    Args:
        connection: A connection object representing the database connection
        cursor: A cursor object to execute database queries
        symbol (str): A string representing the stock symbol.
        start_date (str): A string representing the start date of the range.
        end_date (str): A string representing the end date of the range.
        measured_variable (str): The type of data to measure (e.g., "Sentiments").
        source_type (str, optional): The source type filter. Defaults to 'All'.

    Returns:
        tuple or None: A tuple containing the stock symbol, positive_mentions, negative_mentions, neutral_mentions, and total_mentions or None if no data found.
    """
    
    where_conditions = ["m.date >= %s", "m.date <= %s"]
    parameters = [start_date, end_date]

    # Adding conditions for symbols
    if symbol != "All":
        where_conditions.append("m.symbol = %s")
        parameters.append(symbol)
        
    # Adding conditions for source_type
    if source_type != "All":
        where_conditions.append("m.source_type = %s")
        parameters.append(source_type)
    
    # Constructing WHERE clause
    where_clause = ' AND '.join(where_conditions)

    # Query for sentiments
    if measured_variable == "Sentiments":
        query = f"""
            SELECT
                m.symbol,
                SUM(CASE WHEN ss.sentiment = 'positive' THEN 1 ELSE 0 END) AS positive_mentions,
                SUM(CASE WHEN ss.sentiment = 'negative' THEN 1 ELSE 0 END) AS negative_mentions,
                SUM(CASE WHEN ss.sentiment = 'neutral' THEN 1 ELSE 0 END) AS neutral_mentions,
                COUNT(ss.sentiments_id) AS total_mentions
            FROM mentions m
            JOIN stock_sentiments ss ON m.mention_id = ss.sentiments_id
            WHERE {where_clause}
            GROUP BY m.symbol;
        """
    cursor.execute(query, tuple(parameters))
    result = cursor.fetchone()
    return result


def compare_sentiments_all_stocks_date_range(connection, cursor, start_date, end_date):
    """
    Query for the positive, negative, and neutral sentiments for all stocks on the given date range.
    Also gets the total amount of mentions for each stock.

    Args:
        connection: A connection object representing the database connection
        cursor: A cursor object to execute database queries
        dates (tuple): A tuple containing two date strings representing the start and end date of the range

    Returns:
        list: A list of dictionaries containing the stock symbol, positive_mentions, negative_mentions, neutral_mentions, and total_mentions
    """
    
    query = """
        SELECT
            m.symbol,
            SUM(CASE WHEN ss.sentiment = 'positive' THEN 1 ELSE 0 END) AS positive_mentions,
            SUM(CASE WHEN ss.sentiment = 'negative' THEN 1 ELSE 0 END) AS negative_mentions,
            SUM(CASE WHEN ss.sentiment = 'neutral' THEN 1 ELSE 0 END) AS neutral_mentions,
            COUNT(ss.sentiments_id) AS total_mentions
        FROM mentions m
        JOIN stock_sentiments ss ON m.mention_id = ss.sentiments_id
        WHERE m.date >= %s AND m.date <= %s
        GROUP BY m.symbol;
    """
    cursor.execute(query, (start_date, end_date))
    result = cursor.fetchall()
    return result


def compare_sentiments_data_by_date(connection, cursor, symbol, dates):
    """
    Query for the positive, negative, and neutral sentiments for a given stock on every date entry in the given date range
    Also gets the total amount of mentions for the stock.

    Args:
        connection: A connection object representing the database connection
        cursor: A cursor object to execute database queries
        symbol (str): A string representing the stock symbol.
        dates (tuple): A tuple containing two date strings representing the start and end date of the range

    Returns:
        list: A list of dictionaries containing the stock symbol, date, positive_mentions, negative_mentions, neutral_mentions, and total_mentions
    """
    
    query = """
        SELECT
            m.date,
            m.symbol,
            SUM(CASE WHEN ss.sentiment = 'positive' THEN 1 ELSE 0 END) AS positive_mentions,
            SUM(CASE WHEN ss.sentiment = 'negative' THEN 1 ELSE 0 END) AS negative_mentions,
            SUM(CASE WHEN ss.sentiment = 'neutral' THEN 1 ELSE 0 END) AS neutral_mentions,
            COUNT(ss.sentiments_id) AS total_mentions
        FROM mentions m
        JOIN stock_sentiments ss ON m.mention_id = ss.sentiments_id
        WHERE m.symbol = %s AND m.date >= %s AND m.date <= %s
        GROUP BY m.date
        ORDER BY m.date;
    """
    start_date, end_date = dates[0], dates[1]
    cursor.execute(query, (symbol, start_date, end_date))
    result = cursor.fetchall()
    return result
