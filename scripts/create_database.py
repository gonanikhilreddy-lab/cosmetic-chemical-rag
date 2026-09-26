import duckdb

from src.config.settings import CSV_PATH, DATABASE_PATH


DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
csv_path = str(CSV_PATH).replace("\\", "/").replace("'", "''")

with duckdb.connect(str(DATABASE_PATH)) as connection:
    connection.execute(f"""
        CREATE OR REPLACE TABLE cosmetics AS
        SELECT * FROM read_csv_auto('{csv_path}', header=True)
    """)
    row_count, product_count = connection.execute("""
        SELECT COUNT(*), COUNT(DISTINCT CDPHId) FROM cosmetics
    """).fetchone()

print(f"Rows loaded: {row_count:,}")
print(f"Unique products: {product_count:,}")