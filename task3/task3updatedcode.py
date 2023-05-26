import lmql
import asyncio
import json
import os
import sys

async def query(prompt):
    '''takes in lmql prompt, returns model output'''
    output = (await lmql.run(prompt, output_writer=lmql.stream("RESPONSE")))
    return output

async def main():
    # opening json file
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    codes=data["codes"]

    # getting total accuracy rate of questions from a quiz
    a = 0
    b = 0
    for c in codes:
            response = await query(c["code"])
            answer_from_Output = response[0].variables['ANSWER']
            if answer_from_Output == ("\n\n" + c["answer"]):
                a +=1
            b += 1
    accu_rate = a/b

    # output 
    print(accu_rate)

if __name__ == "__main__":
    asyncio.run(main())
