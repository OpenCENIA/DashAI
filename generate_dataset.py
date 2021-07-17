import csv, json

csvFilePath = "rawTwitterDataset.csv"
jsonFilePath = "twitterDataset.json"

data = {
    'train': [],
    'val': [],
    'test': []
}

with open(csvFilePath) as csvFile:
    csvReader = csv.DictReader(csvFile)
    part = 0
    for row in csvReader:
        actual_row = [int(row["0"]), row["@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D"]]
        actual_data = [{'text': actual_row[1], 'label': actual_row[0]}]
        if part == 0:
            data['train'] += actual_data
        elif part == 1:
            data['val'] += actual_data
        else:
            data['test'] += actual_data
        part = (part + 1)% 3

with open(jsonFilePath, 'w') as jsonFile:
    jsonFile.write(json.dumps(data, indent=4))