import psycopg2
import psycopg2.extras
import os
import logging

# Read environment variables for PostgreSQL connection
# Use standard PostgreSQL environment variable names
dbhost = os.getenv('PGHOST')
dbport = os.getenv('PGPORT')
dbname = os.getenv('PGDATABASE')
dbuser = os.getenv('PGUSER')
dbpass = os.getenv('PGPASSWORD')

def get_connection():
    con = psycopg2.connect(host=dbhost, port=dbport, dbname=dbname, user=dbuser, password=dbpass)
    psycopg2.extras.register_uuid(conn_or_curs=con)
    return con

def find(query, param=None):
    res = None
    con = None
    cur = None
    try:
        con = get_connection()
        cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor) # Use DictCursor here
        if param is None:
            cur.execute(query)
        else:
            cur.execute(query, param)
        res = cur.fetchone()
    except Exception as e:
        logging.error("Database error executing query: %s", str(e))
        logging.error("Query: %s", query)
        logging.error("Params: %s", param)
        raise  # Re-raise the exception so calling code can handle it
    finally:
        if cur: cur.close()
        if con: con.close()
        return res

def execute(query, param=None):
    con = None
    try:
        con = get_connection()
        cur = con.cursor()
        if param is None:
            cur.execute(query)
        else:
            cur.execute(query, param)
        con.commit()
    except Exception as e:
        logging.error("Database error executing query: %s", str(e))
        logging.error("Query: %s", query)
        logging.error("Params: %s", param)
        raise  # Re-raise the exception so calling code can handle it
    finally:
        if con: con.close()
        logging.debug("done")

def find_all(query, param=None):
    res = []
    con = None
    cur = None
    try:
        con = get_connection()
        cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)  # Use DictCursor here
        if param is None:
            cur.execute(query)
        else:
            cur.execute(query, param)
        res = cur.fetchall()
    except Exception as e:
        logging.error("Database error executing query: %s", str(e))
        logging.error("Query: %s", query)
        logging.error("Params: %s", param)
        raise  # Re-raise the exception so calling code can handle it
    finally:
        if cur: cur.close()
        if con: con.close()
        return res