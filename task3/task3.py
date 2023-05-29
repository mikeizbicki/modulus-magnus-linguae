import lmql
import asyncio
import json
import os
import sys
import csv

async def query(prompt):
    '''takes in lmql prompt, returns model output'''
    output = (await lmql.run(prompt, output_writer=lmql.stream("RESPONSE")))
    return output

async def main():
    # opening json file
    # TODO: edit to be more generalized
    jsonName = sys.argv[1]
    with open(jsonName, 'r') as f:
        data = json.load(f)
    codes=data["codes"]

    # getting total accuracy rate of questions from a quiz
    correct = 0
    for c in codes:
            response = await query(c["code"])
            answer_from_Output = response[0].variables['ANSWER']
            if answer_from_Output == ("\n\n" + c["answer"]):
                correct +=1
    accu_rate = correct/len(codes)

    # output 
    print(accu_rate)
    toCSV(jsonName, accu_rate)

def toCSV(jsonName, accu_rate):
    '''adds info to quiz type csv'''
    #TODO: get from folder name
    quiztype = "test"
    
    f = open('test.txt', 'w+') 
    f.write('python rules')
    f.close()    
    
    infoList = jsonName.split("_")
    # add all of that info + acc_rate to a new line in the csv file
    with open("quiztype" + quiztype + ".csv", 'w+') as f:
        writer = csv.writer(f)
        # write to csv with columns as:  model, quiz chapter, prompt style, and accuracy rate
        writer.writerow([infoList[2].replace(".json",""), infoList[0], infoList[1], accu_rate])
        f.flush()

    print("dne?")

if __name__ == "__main__":
    asyncio.run(main())
