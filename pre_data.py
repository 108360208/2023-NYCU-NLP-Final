import csv
import json

# Open the CSV file
with open('/home/fansa/NLP_vae/dataset/CVAT_4_SD.csv', 'r', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file, delimiter='\t')

    # Create a list to store the JSON objects
    json_objects = []

    # Iterate over each row in the CSV file
    for row in reader:
        # Extract the necessary information from the row
        print(row)
        instruction = "test"
        output = []
        output.append(row['Valence_Mean'])
        output.append(row['Arousal_Mean'])
        
        # Create a dictionary for the JSON object
        json_object = {
            "instruction": instruction,
            "input": row['Text'],
            "output": output
        }
      
        # Append the JSON object to the list
        json_objects.append(json_object)

# Write the JSON objects to a file
with open('output.json', 'w') as json_file:
    json.dump(json_objects, json_file, indent=4, ensure_ascii=False)