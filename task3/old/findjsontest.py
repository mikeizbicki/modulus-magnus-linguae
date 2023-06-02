import json
import os
import asyncio
file_name = '/home/asawyer/summer23/modulus-magnus-linguae/task3/sample_answers_Task3.json'
# Open the JSON file
with open(file_name, 'r') as f:
    data = json.load(f)

print(data)
print(os.getcwd())
