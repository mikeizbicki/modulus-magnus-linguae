import lmql
import asyncio
  
@lmql.query
async def question():
    '''lmql
    argmax"answer this latin Brundisium et Tusculum oppid~ ~. [ANSWER] " from "openai/text-ada-001"
    '''

async def main():
    output = (await question())[0].prompt
    print(output)

# "main class"
if __name__ == "__main__":
    asyncio.run(main())

