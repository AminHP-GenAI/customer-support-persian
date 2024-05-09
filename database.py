import os
import shutil
import requests
import sqlite3
import pandas as pd


class Database:

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.download()
        self.reset_and_prepare()

    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, 'travel.sqlite')

    @property
    def db_backup_path(self) -> str:
        return os.path.join(self.data_dir, 'travel.backup.sqlite')

    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def download(self, overwrite: bool = False) -> None:
        if not overwrite and os.path.exists(self.db_backup_path):
            return

        db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
        response = requests.get(db_url)
        response.raise_for_status()

        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.db_backup_path, 'wb') as f:
            f.write(response.content)

    def reset_and_prepare(self) -> None:
        shutil.copy(self.db_backup_path, self.db_path)

        connection = self.get_connection()

        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table';", connection
        ).name.tolist()
        tdf = {}
        for t in tables:
            tdf[t] = pd.read_sql(f"SELECT * from {t}", connection)

        example_time = pd.to_datetime(
            tdf['flights']['actual_departure'].replace('\\N', pd.NaT)
        ).max()
        current_time = pd.to_datetime('now').tz_localize(example_time.tz)
        time_diff = current_time - example_time

        tdf['bookings']['book_date'] = (
            pd.to_datetime(tdf['bookings']['book_date'].replace('\\N', pd.NaT), utc=True)
            + time_diff
        )

        datetime_columns = [
            'scheduled_departure',
            'scheduled_arrival',
            'actual_departure',
            'actual_arrival',
        ]
        for column in datetime_columns:
            tdf['flights'][column] = (
                pd.to_datetime(tdf['flights'][column].replace('\\N', pd.NaT)) + time_diff
            )

        for table_name, df in tdf.items():
            df.to_sql(table_name, connection, if_exists='replace', index=False)

        connection.commit()
        connection.close()
