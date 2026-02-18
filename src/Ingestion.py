import duckdb
import os
import src.path_finder as pathfinder


######### Eddie Part

class DataIngestion:
    def __init__(self, data_path, custom_db_path=None):
        self.data_path = data_path
        if not os.path.exists(pathfinder.DATABASE_DIR):
            os.makedirs(pathfinder.DATABASE_DIR)
        self.db_path = os.path.join(pathfinder.DATABASE_DIR, "CRMLS.db") if not custom_db_path else custom_db_path
        if not os.path.exists(self.db_path):
            self.files = self.search_files()
            self.init_db()

    def init_db(self):
        with duckdb.connect(self.db_path) as conn:
            if self.files is None:
                raise ValueError('No valid input CSV files found')

            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS Property AS
                SELECT
                    * EXCLUDE (filename),
                    strptime(regexp_extract(filename, 'CRMLSSold(\\d{{6}})\\.csv$', 1), '%Y%m')::DATE AS ReadDate
                FROM read_csv_auto('{os.path.join(pathfinder.CSV_DIR, self.files[0])}', filename = True, nullstr = ['NaN','nan','N/A','NA','', 'NULL'])
                """
            )
            if len(self.files) > 1:
                self.insert_data(self.files[1:])

    def insert_data(self, files: list):

        with duckdb.connect(self.db_path) as conn:
            for n in range(len(files)):
                conn.execute(
                    f"""
                    INSERT INTO Property
                    SELECT
                        * EXCLUDE (filename),
                        strptime(regexp_extract(filename, 'CRMLSSold(\\d{{6}})\\.csv$', 1), '%Y%m')::DATE AS ReadDate
                    FROM read_csv_auto('{os.path.join(pathfinder.CSV_DIR, files[n])}', filename = True, nullstr = ['NaN','nan','N/A','NA','', 'NULL'])
                    """
                )

    def search_files(self):
        files = [f for f in os.listdir(self.data_path) if not f.startswith('.')]
        return files if len(files) > 0 else None

    def query(self, query_cmd, values=None):
        with duckdb.connect(self.db_path) as conn:
            if values is None:
                conn.sql(query_cmd)
            else:
                conn.sql(query_cmd, params=values)

            if "SELECT" in query_cmd:
                return conn.query(query_cmd).to_df()
            return None

    def export_csv(self, query, custom_name: str, custom_output_path=''):
        if not os.path.exists(pathfinder.OUTPUT_DIR):
            os.makedirs(pathfinder.OUTPUT_DIR)
        with duckdb.connect(self.db_path) as conn:
            data = conn.execute(query).fetchdf()
            if custom_output_path != '':
                data.to_csv(custom_output_path)
            data.to_csv(os.path.join(pathfinder.OUTPUT_DIR, custom_name) + ".csv")

