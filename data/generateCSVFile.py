import json
import csv
import os

jsonl_file = input("Enter the name of the jsonl file: ")
csv_file = 'MergeFile.csv'

language = input("Enter the language for this dataset: ")

keys = ['prompt', 'human_text', 'machine_text', 'model', 'source', 'language']

file_exists = os.path.isfile(csv_file)

with open(jsonl_file, 'r', encoding='utf-8') as infile, open(csv_file, 'a', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=keys)

    if not file_exists or os.stat(csv_file).st_size == 0:
        writer.writeheader()

    for line in infile:
        if not line.strip():
            continue
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(f"Warning: Skipping malformed JSON line: {line.strip()}")
            continue

        row = {key: data.get(key, '-') for key in keys if key != 'language'}

        row['language'] = language

        writer.writerow(row)

print(f"Data appended to CSV file '{csv_file}' with language '{language}' successfully!")
