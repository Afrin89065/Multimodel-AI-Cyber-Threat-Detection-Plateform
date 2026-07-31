import json

train = set()

with open("datasets/processed/fusion/train.jsonl") as f:
    for line in f:
        train.add(str(json.loads(line)["features"]))

count = 0

with open("datasets/processed/fusion/test.jsonl") as f:
    for line in f:
        if str(json.loads(line)["features"]) in train:
            count += 1

print("Train/Test overlap:", count)