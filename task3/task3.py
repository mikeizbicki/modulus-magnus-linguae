import os
import lmql
import asyncio
import json
import os
import sys
import csv
from  pathlib import Path

async def query(c):
    '''takes in lmql prompt, returns array of test results'''
    correctAnswer = "\n\n" + c["answer"]
    output = (await lmql.run(c["code"], output_writer=lmql.stream("RESPONSE")))
    if output[0].variables['ANSWER'] == correctAnswer:
        return 1
    return 0

async def main():
    # printing csv column names: model, quiz chapter, prompt style, and accuracy rate
    print("model, quizCh, promptStyle, accuracy")

    # opening json file
    directory = sys.argv[1]
    for file_path in (Path.cwd()/directory).glob("*.json"):

        with file_path.open(mode='r') as f:
            data = json.load(f)
        codes=data["codes"]

        # getting total accuracy rate of questions from a quiz
        results = await asyncio.gather(*[query(c) for c in codes])
        accuracy = sum(results)/len(results)
    
        # parse json file
        infoList = file_path.stem.split(".")
    
        # printing row of csv file
        output = [infoList[2], infoList[0], infoList[1], str(accuracy)]
        print(", ".join(output))

if __name__ == "__main__":
    asyncio.run(main())
