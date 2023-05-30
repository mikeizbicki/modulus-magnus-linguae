import lmql
import asyncio
import json
import os
import sys
import csv
from  pathlib import Path

async def query(c):
    '''takes in a dict that includes lmql prompt and the true answer,
    returns 1 if the model is correctand 0 if wrong'''
    # formating answer to use to test exact match
    correctAnswer = "\n\n" + c["answer"]

    # querying lmql 
    output = (await lmql.run(c["code"], output_writer=lmql.stream("RESPONSE")))
    if output[0].variables['ANSWER'] == correctAnswer:
        return 1
    return 0

def calcAccuracy(data):
    '''calculates accuracy based on outputs from query'''
    codes=data["codes"]
    
    # using query to prompt model with questions in parallel 
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*[query(c) for c in codes]))
    loop.close()

    return sum(results)/len(results)

def main():
    # printing csv column names
    print("model, quizCh, promptStyle, accuracy")

    # opening json file
    directory = sys.argv[1]
    for file_path in (Path.cwd()/directory).glob("*.json"):
        with file_path.open(mode='r') as f:
            data = json.load(f)

        # printing string of what will be a row of the csv file
        infoList = file_path.stem.split(".")
        output = [infoList[2], infoList[0], infoList[1], str(calcAccuracy(data))]
        print(", ".join(output))

if __name__ == "__main__":
    main()
