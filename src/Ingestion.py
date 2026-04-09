import logging

import duckdb
import os
import src.path_finder as pathfinder


######### Eddie Part

class DataIngestion:
    def __init__(self, data_path=pathfinder.DATA_DIR, custom_db_path=None):
        self.data_path = data_path
        print('Creating database directory...')
        os.makedirs(pathfinder.DATABASE_DIR,exist_ok=True)
        os.makedirs(pathfinder.ARTIFACTS_DIR,exist_ok=True)
        print('Data folders created.')
        self.db_path = os.path.join(pathfinder.DATABASE_DIR, "CRMLS.db") if not custom_db_path else custom_db_path

        ## IF there is no database file, create one. ELSE, look for new files and initialize the connection
        if not os.path.exists(self.db_path):
            print('No database file found. Creating a new one...')
            self.files = self.search_new_files(init=True)
            self.create_database()
            print('Database file created. Connection initialized.')
        else:
            print('Database file found. Initializing connection...')
            self.files = self.search_new_files(init=False)
            self.init_db()
            print('Connection initialized.')

    def init_db(self):
        """
        This module provides a way to interact with a DuckDB database.
        It allows connecting to the database, import CSV files into it.
        """
        with duckdb.connect(self.db_path) as conn:
            if self.files is None :
                print('No new CSVs to import to the database. Returning connection...')
            else:
                print(f'Found {len(self.files)} new file(s). Importing to the database...')
                self.insert_data(self.files)

    def insert_data(self, files: list):
        """
        This function inserts data from CSV files into the 'Property' table in a DuckDB database.

        :param files: A list of file paths to CSV files.
        :type files: list
        :raises Exception: If any error occurs during the database operation.
        """
        with duckdb.connect(self.db_path) as conn:
            for n in range(len(files)):
                print(f'Importing {files[n]} ({n+1}/{len(files)})')
                conn.execute(
                    f"""
                    INSERT INTO Property
                    SELECT
                        * EXCLUDE (filename),
                        strptime(regexp_extract(filename, 'CRMLSSold(\\d{{6}})\\.csv$', 1), '%Y%m')::DATE AS ReadDate
                    FROM read_csv_auto('{os.path.join(pathfinder.CSV_DIR, files[n])}', filename = True, nullstr = ['NaN','nan','N/A','NA','', 'NULL'])
                    """
                )

    def create_database(self):
        """
        Creates a database table and a view based on the CSV file.

        This method connects to the database specified in the `db_path`
        attribute and executes SQL queries to create a table named 'Property'
        and a view named 'InsertedFiles'. The 'Property' table is populated
        with data from the CSV file, extracting relevant information such as
        file names and reading dates. The 'InsertedFiles' view provides a
        distinct list of file dates sorted by reading date in descending order.
        """
        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                f"""
                    CREATE TABLE IF NOT EXISTS Property AS
                    SELECT
                        * EXCLUDE (filename),
                        strptime(regexp_extract(filename, 'CRMLSSold(\\d{{6}})\\.csv$', 1), '%Y%m')::DATE AS ReadDate
                    FROM read_csv_auto('{os.path.join(pathfinder.CSV_DIR, self.files[0])}', filename = True, nullstr = ['NaN','nan','N/A','NA','', 'NULL'])
                """
            )

            conn.execute(
                f"""
                    CREATE VIEW InsertedFiles AS
                    SELECT DISTINCT strftime('CRMLSSold%Y%m.csv', ReadDate) AS FileDate
                    FROM Property
                    ORDER BY ReadDate DESC
                """
            )

            if len(self.files) > 1:
                self.insert_data(self.files[1:])

    def search_new_files(self, init:bool):
        """
        This function searches for new files within a specified directory
        and compares them against a database of previously inserted files.

        Args:
            init (bool): A boolean flag indicating whether the file insertion
                          has been initialized. If True, the function skips
                          the database comparison.

        Returns:
            list: A list of filenames that are present in the directory but
                  not in the database, or None if no new files are found.
        """
        files = [f for f in os.listdir(self.data_path) if not f.startswith('.')]
        if not init:
            with duckdb.connect(self.db_path) as conn:
                inserted_files = conn.execute(
                    """
                    SELECT * FROM InsertedFiles
                    """
                ).fetchall()
                inserted_files = [item[0] for item in inserted_files]
                files = list(set(files) - set(inserted_files))
        return files if len(files) > 0 else None

    def query(self, query_cmd, values=None):
        """Executes a SQL query against the database.

        This method executes a given SQL query command using the DuckDB connection.
        It handles both cases where no values are provided and where values are
        passed as a parameter to the query.  If the query command starts with
        "SELECT", the results are converted into a Pandas DataFrame.

        Args:
            query_cmd (str): The SQL query command to execute.
            values (list, optional): A list of values to be used as parameters
                in the query. Defaults to None.

        Returns:
            pd.DataFrame | None: A Pandas DataFrame containing the results of the
                query if the query command starts with "SELECT". Otherwise,
                returns None.
        """
        with duckdb.connect(self.db_path) as conn:
            if values is None:
                conn.sql(query_cmd)
            else:
                conn.sql(query_cmd, params=values)

            if "SELECT" in query_cmd:
                df = conn.query(query_cmd).to_df()
                print(f'Retrieved {df.shape[0]} rows of data from the database.')
                return df
            return None

    def export_csv(self, query, custom_name: str, custom_output_path=''):
        if not os.path.exists(pathfinder.OUTPUT_DIR):
            os.makedirs(pathfinder.OUTPUT_DIR)
        with duckdb.connect(self.db_path) as conn:
            data = conn.execute(query).fetchdf()
            if custom_output_path != '':
                data.to_csv(custom_output_path)
            data.to_csv(os.path.join(pathfinder.OUTPUT_DIR, custom_name) + ".csv")


if __name__ == '__main__':
    conn = DataIngestion(pathfinder.CSV_DIR)
