import os
from sqlalchemy import create_engine, text  # Import text for executable query
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

def sql_connection():
    """
    Creates and tests a SQLAlchemy engine for a MySQL database connection.
    Loads DB credentials from environment variables.
    Returns the engine if connection is successful.
    Raises an exception if connection fails or environment variables are missing.
    """

    # Load environment variables from .env file
    load_dotenv()

    # Retrieve database credentials and details from environment variables
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))  # Default MySQL port is 3306
    DB_NAME = os.getenv("DB_NAME")
    print(DB_USER ,DB_HOST)

    # Check if any of the critical variables are missing
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
        raise ValueError("One or more database environment variables are missing")

    # Construct the database connection string in SQLAlchemy format
    connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Initialize the SQLAlchemy engine
    engine = create_engine(connection_string)

    try:
        # Attempt to connect to the database and execute a simple test query
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        # If successful, return the engine object
        return engine

    except OperationalError as e:
        # Raise an error if connection fails, including details for debugging
        raise ConnectionError(f"Database connection failed: {e}")
# if __name__ == "__main__":
#     try:
#         engine = my_sql_connection()
#         print("Database connection successful.")
#     except Exception as e:
#         print(f"Error: {e}")