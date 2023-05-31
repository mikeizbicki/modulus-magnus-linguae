import lmql
import asyncio
import json
import os
import sys
import csv
from pathlib import Path
import argparse

async def query(c):
    '''takes in a dict that includes lmql prompt and the true answer,
    returns 1 if the model is correctand 0 if wrong'''
    output = (await lmql.run(c["code"], output_writer=lmql.stream("RESPONSE")))
    if output[0].variables['ANSWER'] == c["answer"]:
        return 1
    return 0

def calcAccuracy(codes):
    '''calculates accuracy based on outputs from query

    >>> calcAccuracy([{'code': ('argmax "Q: Fill in the missing words: Italia ~ Europa est; Graecia ~ in Europa est. '
    ...                 'Answer Choices: (A) in quoque (B) ne non (C) ubi (D) non sed '
    ...                 'A: [ANSWER]" from "openai/text-davinci-003" where ANSWER in ["A", "B","C", "D"]'),'answer': 'A'}, {'code': ('argmax' 
    ...                 '"Q: Fill in the missing words: ~ est Arabia? In Asia est Arabia. '
    ...                 'Answer Choices: (A) in quoque (B) ne non (C) ubi (D) non sed '
    ...                 'A: [ANSWER]" from "openai/text-davinci-003" where ANSWER in ["A", "B","C", "D"]'),'answer':'C'}])
    0.5

    >>> calcAccuracy([{'code': ('argmax "Q: Fill in the tilde: Italia in Europa ~. '
    ...                 'Answer Choices: (A) est (B) sunt '
    ...                 'A: [ANSWER]" from "openai/text-davinci-003" where ANSWER in ["A", "B"]'),'answer': 'A'}, {'code': ('argmax'
    ...                 '"Q: Fill in the tilde: Italia et Gallia in Europa ~. '
    ...                 'Answer Choices: (A) est (B) sunt '
    ...                 'A: [ANSWER]" from "openai/text-davinci-003" where ANSWER in ["A", "B"]'),'answer':'B'}])
    1.0

    '''
    # using query to prompt model with questions in parallel 
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*[query(c) for c in codes]))

    return round(sum(results)/len(results), 2)

def main():
    # printing csv column names
    print("model, quizCh, promptStyle, accuracy")

    # opening json file
    parser = argparse.ArgumentParser()
    parser.add_argument('second_argument')

    for file_path in (Path.cwd()/parser.parse_args().second_argument).glob("*.json"):
        with file_path.open(mode='r',encoding="utf-8") as f:
            data = json.load(f)

        # printing string of what will be a row of the csv file
        infoList = file_path.stem.split(".")
        output = [infoList[2], infoList[0], infoList[1], str(calcAccuracy(data["codes"]))]
        print(", ".join(output))

if __name__ == "__main__":
    main()
