import lmql

response = await lmql.run('''argmax"answer this latin Brundisium et Tusculum oppid~ ~. [ANSWER] " from "openai/text-ada-001"''', output_writer=lmql.stream("RESPONSE"))

prompt = response[0].prompt
answer = response[0].variables['ANSWER']

print(prompt,answer)
