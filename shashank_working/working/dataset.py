import pandas as pd

def process_traffic_data(traffic_csv, latlong_csv, output_csv):
    # Load the datasets
    try:
        traffic_df = pd.read_csv(traffic_csv)
        latlong_df = pd.read_csv(latlong_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Keep only the first occurrence for each Area Name in the traffic dataset
    # This addresses the requirement to take the first value for multiple entries
    traffic_unique = traffic_df.drop_duplicates(subset=['Road/Intersection Name'], keep='first')

    # Merge the dataframes
    # We join 'Area Name' from traffic_df with 'name' from latlong_df
    merged_df = pd.merge(
        traffic_unique, 
        latlong_df, 
        left_on='Road/Intersection Name', 
        right_on='name', 
        how='inner'
    )

    # Select the required columns
    result_df = merged_df[[
        'Area Name', 
        'latitude', 
        'longitude', 
        'Traffic Volume', 
        'Road Capacity Utilization'
    ]]

    # Save to the 3rd CSV file
    result_df.to_csv(output_csv, index=False)
    print(f"Successfully created '{output_csv}' with {len(result_df)} rows.")

# Execute the processing
process_traffic_data('Bangalore_Traffic_Dataset.csv', 'latlong.csv', 'output.csv')