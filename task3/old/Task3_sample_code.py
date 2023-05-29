import json

list_object = [
    "Q: Nilus fluvi~ est. A:",
    "us"
]

file_name = 'sample_answers_Task3.json'

def match_test(list_object, file_name):
    # Open the JSON file
    with open(file_name, 'r') as f:
        data = json.load(f)

    questions = data["questions"]

    for q in questions:
        if list_object[0] in q["prompts"]:
            # If the second object in list_object matches the "answer"
            if list_object[1] == q["answer"]:
                return 1

    # If no matches were found, return 0
    return 0
print(match_test(list_object, file_name))
