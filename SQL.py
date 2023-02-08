import psycopg2
import database_config


##create tables in the PostgreSQL database if they do not exist
def create_tables(connection,cursor):
    tables = (
        """
        CREATE TABLE IF NOT EXISTS stock_info (
            entry_id SERIAL PRIMARY KEY,
            date VARCHAR(255) NOT NULL,
            symbol VARCHAR(6) NOT NULL,
            low NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            mentions INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS mentions (
                mention_id SERIAL PRIMARY KEY,
                symbol VARCHAR(6) NOT NULL,
                sentiment VARCHAR(8) NOT NULL,
                mention_text TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT NOT NULL
        )
        """)
    try:
        for table in tables:
            cursor.execute(table)
            
        print("Tables created/validated successfully")
    except Exception as err:
        print("failed to create tables")
        print(f"Error: '{err}'")
        connection.rollback()
        
## Create the postgres database if it does not exist
## Autocommit is enabled because you cannot create a database otherwise
def create_database(connection, cursor):
    try:
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'hype_stock'")
        exists = cursor.fetchone()
        if not exists:
            cursor.execute('CREATE DATABASE hype_stock')
            print("Database created successfully")
        else:
            print("Database exists already")
    except Exception as err:
        print("failed to create database")
        print(f"Error: '{err}'")
        connection.rollback()

def insert_stock(connection, cursor, stock_data):
    try:
        time, symbol, low, high = stock_data[0], stock_data[1], stock_data[2], stock_data[3]   
        cursor.execute("""
            INSERT INTO stock_info (date, symbol, low, high, mentions)
            VALUES (%s, %s, %s, %s, %s)""", (time, symbol, low, high,0))
        
        print("Stock search info inserted into database")
    except Exception as err:
        print("failed to insert stock into database")
        print(f"Error: '{err}'")
        connection.rollback()
        
def insert_mention(connection, cursor, stock_data, sentiment, text, url):
    try:
        time, symbol, low, high = stock_data[0], stock_data[1], stock_data[2], stock_data[3]   
        cursor.execute("""
            INSERT INTO mentions (symbol, sentiment, mention_text, source, url)
            VALUES (%s, %s, %s, 'wallstreetbets', %s)""", (symbol, sentiment ,text, url))
        
        print("Stock mention info inserted into database")
    except Exception as err:
        print("failed to insert stock mention into database")
        print(f"Error: '{err}'")
        connection.rollback()
