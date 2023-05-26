import lmql
import asyncio
import json
import os

async def query(prompt):
    output = (await lmql.run(prompt, output_writer=lmql.stream("RESPONSE")))
    return output

def match_test(response, file_name):
    # Open the JSON file
    with open(file_name, 'r') as f:
        data = json.load(f)

    questions = data["questions"]

    for q in questions:
      for question in q["prompts"]:
        if question in prompt:
            if answer == q["answer"]:
                return 1
    return 0

if __name__ == "__main__":
    testPrompt = '''argmax"Q: Nilus fluvi~ est. A:[ANSWER]" from "openai/text-ada-001"'''
    response = asyncio.run(query(testPrompt))
    prompt = response[0].prompt
    answer = response[0].variables['ANSWER']
    print(prompt,answer)
   # file_name = '/home/asawyer/summer23/modulus-magnus-linguae/task3/sample_answers_Task3.json'
   # print(match_test(response, file_name))
