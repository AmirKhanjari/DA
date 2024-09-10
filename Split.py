import pandas as pd

# Updated paths
CSV_FILE1 = '/home/amirkh/Python/Main/CSV/Fin1-3(6).csv'
CSV_FILE2 = '/home/amirkh/Python/Main/CSV/Fin2(6).csv'

# Read the CSV files
df1 = pd.read_csv(CSV_FILE1)
df2 = pd.read_csv(CSV_FILE2)

# Get unique labels from both datasets
labels1 = set(df1['taxon'].unique())
labels2 = set(df2['taxon'].unique())

# Calculate overlapped labels
overlapped_labels = labels1.intersection(labels2)

# Print results
print(f"Number of unique labels in Fin1: {len(labels1)}")
print(f"Number of unique labels in Fin2: {len(labels2)}")
print(f"Number of overlapped labels: {len(overlapped_labels)}")
print("\nOverlapped labels:")
for label in sorted(overlapped_labels):
    print(f"- {label}")

# Print labels unique to each dataset
print("\nLabels unique to Fin1:")
for label in sorted(labels1 - labels2):
    print(f"- {label}")

print("\nLabels unique to Fin2:")
for label in sorted(labels2 - labels1):
    print(f"- {label}")