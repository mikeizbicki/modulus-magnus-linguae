from xml.etree import cElementTree as ET
import os
import string 
import tiktoken
# encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


# input file 
os.chdir("/Users/aly/Desktop/CS/Research/Summer23")
tree = ET.parse("Achuar-NT.xml")
root = tree.getroot()

# putting all items into array
arr=[]
for item in root.findall('.//w'):
    arr.append(item.text)

# getting tokens
sum = 0
for word in arr:
    sum += len(word.encode())
print(sum)

