import json
import os

# Path to your JSON transcript
src = "doc/transcripts/116_P8_conversation.json"
dst = "doc/transcripts/116_P8_conversation.txt"

with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)

# Flatten text
lines = []
if isinstance(data, dict) and "full_conversation" in data:
    for turn in data["full_conversation"]:
        lines.append(turn)
elif isinstance(data, list):
    for item in data:
        lines.append(str(item))

# Write as readable text
with open(dst, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"✅ Saved text transcript to {dst}")
