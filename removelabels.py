import pandas as pd
import os

# Updated paths
CSV_FILE1 = '/home/amirkh/Python/Main/CSV/01_finbenthic1-1_processed_5splits_taxon.csv'
CSV_FILE2 = '/home/amirkh/Python/Main/CSV/01_finbenthic2_processed_5splits_taxon.csv'
CSV_FOLDER = '/home/amirkh/Python/Main/CSV/'

# Read the CSV files
df1 = pd.read_csv(CSV_FILE1)
df2 = pd.read_csv(CSV_FILE2)

# Get unique labels from both datasets
labels1 = set(df1['taxon'].unique())
labels2 = set(df2['taxon'].unique())

# Calculate unique labels in CSV_FILE1
unique_labels_csv1 = labels1 - labels2

# Filter df1 to keep only rows with unique taxon
df_unique = df1[df1['taxon'].isin(unique_labels_csv1)]

# Create a new CSV file for unique taxon from CSV_FILE1
new_csv_file = os.path.join(CSV_FOLDER, 'unique_taxon_csv1.csv')
df_unique.to_csv(new_csv_file, index=False)

# Print results
print(f"Number of unique labels in Fin1: {len(labels1)}")
print(f"Number of unique labels in Fin2: {len(labels2)}")
print(f"Number of overlapped labels: {len(labels1.intersection(labels2))}")
print(f"Number of unique labels in CSV_FILE1: {len(unique_labels_csv1)}")
print(f"\nUnique labels from CSV_FILE1 have been saved to: {new_csv_file}")

# Print some unique labels as an example
print("\nExample of unique labels from CSV_FILE1:")
for label in sorted(unique_labels_csv1)[:10]:  # Print first 10 as an example
    print(f"- {label}")