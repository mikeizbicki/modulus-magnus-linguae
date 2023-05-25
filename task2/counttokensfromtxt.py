import tiktoken
import os
from pathlib import Path

# encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

os.chdir("/Users/aly/Desktop/CS/Research/Summer23/txt combined")
txt = Path("acu.txt").read_text()
txt = txt.replace('\n', ' ')

# getting tokens
sum = len(txt.encode())
print(sum)


