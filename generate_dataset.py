import csv, json

csvFilePath = "Example_dataset/IMDB_Sentiment.csv"
jsonFilePath = "Example_dataset/ImdbSentimentDataset.json"

size = 100
every = 4 # Every 'every' train registers, one is assigned to test dataset

data = {
    "task_info":{"task_type":"TextClassificationTask"},
    'train': {'x':[], 'y':[]},
    'test': {'x':[], 'y':[]}
}

with open(csvFilePath, encoding="utf8") as csvFile:
    csvReader = csv.DictReader(csvFile)
    c = 0
    p = 0
    for row in csvReader:
        try:
            actual_data = [row['\ufefftext'], int(row['label'])]
        except:
            continue

        if c%(every+1) != 0:
            data['train']['x'].append(actual_data[0])
            data['train']['y'].append(actual_data[1])

        elif c%(every+1) == 0:
            data['test']['x'].append(actual_data[0])
            data['test']['y'].append(actual_data[1])
        else:
            break
        c += 1

with open(jsonFilePath, 'w') as jsonFile:
    jsonFile.write(json.dumps(data, indent=4))

###############################################################################

# csvFilePath = "Example_dataset/Iris.csv"
# jsonFilePath = "Example_dataset/irisDataset.json"

# size = 100
# every = 4 # Every 'every' train registers, one is assigned to test dataset

# data = {
#     "task_info":{"task_type":"NumericClassificationTask"},
#     'train': {'x':[], 'y':[]},
#     'test': {'x':[], 'y':[]}
# }

# with open(csvFilePath, encoding="utf8") as csvFile:
#     csvReader = csv.DictReader(csvFile)
#     c = 0
#     p = 0
#     for row in csvReader:
#         try:
#             actual_data = [[float(row[i]) for i in ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]], row['Species']]
#         except:
#             continue

#         if actual_data[1] == 0:
#             continue

#         if c%(every+1) != 0:
#             data['train']['x'].append(actual_data[0])
#             data['train']['y'].append(actual_data[1])

#         elif c%(every+1) == 0:
#             data['test']['x'].append(actual_data[0])
#             data['test']['y'].append(actual_data[1])
#         else:
#             break
#         c += 1

# with open(jsonFilePath, 'w') as jsonFile:
#     jsonFile.write(json.dumps(data, indent=4))

###############################################################################

# csvFilePath = "Example_dataset/Twitter_Data.csv"
# jsonFilePath = "Example_dataset/twitterDataset.json"

# size = 1000

# data = {
#     'train': {'x':[], 'y':[]},
#     'test': {'x':[], 'y':[]}
# }

# with open(csvFilePath, encoding="utf8") as csvFile:
#     csvReader = csv.DictReader(csvFile)
#     c = 0
#     p = 0
#     for row in csvReader:
#         try:
#             actual_data = [row['clean_text'], int(row['category'])]
#         except:
#             continue

#         if actual_data[1] == 0:
#             continue

#         if c == size:
#             c = 0
#             p += 1
#         if p == 0 :
#             data['train']['x'].append(actual_data[0])
#             data['train']['y'].append(actual_data[1])
#         elif p == 1:
#             data['test']['x'].append(actual_data[0])
#             data['test']['y'].append(actual_data[1])
#         else:
#             break
#         c += 1

# with open(jsonFilePath, 'w') as jsonFile:
#     jsonFile.write(json.dumps(data, indent=4))