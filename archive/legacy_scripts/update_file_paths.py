import csv

# Input and output CSV file paths
input_csv = 'data\metadata\dataset_meta_v3.csv'  # Change to the correct input CSV file path
output_csv = 'data\metadata\dataset_meta_v3.1.csv'  # Output CSV file path

# Open the CSV file to read data
with open(input_csv, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    
    # Prepare for writing the updated CSV
    fieldnames = reader.fieldnames
    rows = []
    
    # Process each row
    for row in reader:
        # Modify the file_path (replace "raw" with "data/raw")
        row['file_path'] = row['file_path'].replace('raw\\', 'data/raw\\')
        rows.append(row)
        
# Write the updated data to a new CSV
with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated CSV saved as {output_csv}")