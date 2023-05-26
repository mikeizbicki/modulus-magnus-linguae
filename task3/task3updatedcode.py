import lmql
import asyncio
import json
import os
import sys

async def query(prompt):
    output = (await lmql.run(prompt, output_writer=lmql.stream("RESPONSE")))
    return output

async def main(filename):
    codes=data["codes"]
    a = 0
    b = 0
    for c in codes:
            response = await query(c["code"])
            answer_from_Output = response[0].variables['ANSWER']
            if answer_from_Output == ("\n\n" + c["answer"]):
                a +=1
            b += 1
    accu_rate = a/b
    print(accu_rate)

if __name__ == "__main__":
    print(len(sys.argv))
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    asyncio.run(main(data))
