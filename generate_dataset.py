import csv, json

csvFilePath = "Twitter_Data.csv"
jsonFilePath = "twitterDataset.json"

data = {
    'train': [],
    'val': [],
    'test': []
}

with open("Twitter_Data.csv", encoding="utf8") as csvFile:
    csvReader = csv.DictReader(csvFile)
    part = 0
    for row in csvReader:
        actual_data = [{'text': row['clean_text'], 'label': int(row['category'])}]
        if part == 0:
            data['train'] += actual_data
        elif part == 1:
            data['val'] += actual_data
        else:
            data['test'] += actual_data
        part = (part + 1)% 3

with open(jsonFilePath, 'w') as jsonFile:
    jsonFile.write(json.dumps(data, indent=4))