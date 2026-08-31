import duckdb
import pandas as pd
import numpy as np

def load_and_prep_cargo_data(csv_path: str = "data/flights.csv") -> pd.DataFrame:
    #Queries flight data using DuckDB and generates operational cargo metrics.
    con = duckdb.connect()
    
    query = f"""
        SELECT 
            ORIGIN_AIRPORT AS Origin,
            DESTINATION_AIRPORT AS Dest,
            MONTH,
            DAY_OF_WEEK,
            DISTANCE,
            DEPARTURE_DELAY AS DEP_DELAY,
            ARRIVAL_DELAY AS ARR_DELAY,
            CASE WHEN ARRIVAL_DELAY > 15 THEN 1 ELSE 0 END AS delay_flag
        FROM read_csv_auto('{csv_path}')
        WHERE CANCELLED = 0 AND DEPARTURE_DELAY IS NOT NULL
        LIMIT 50000
    """
    df = con.execute(query).df()
    
    # Generate realistic simulated operational payload data based on distance & delays
    np.random.seed(42)
    df['passenger_count'] = np.random.randint(80, 220, size=len(df))
    df['baggage_count'] = (df['passenger_count'] * np.random.uniform(1.1, 1.4, size=len(df))).astype(int)
    
    # Connection tightness factor (simulated minutes between connecting flights)
    df['connect_time_mins'] = np.random.randint(25, 120, size=len(df))
    
    # Target label: Baggage misconnect risk
    df['baggage_misconnect'] = np.where(
        (df['connect_time_mins'] < 45) & (df['DEP_DELAY'] > 10), 1, 0
    )
    
    return df

if __name__ == "__main__":
    data = load_and_prep_cargo_data()
    print("Pipeline Success. Extracted rows:", len(data))