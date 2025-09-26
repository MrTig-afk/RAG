import json
import re

file_path = "truth.md"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Regex to find all JSON arrays in the file
json_arrays = re.findall(r'\[\s*{.*?}\s*\]', content, flags=re.DOTALL)

ground_truth_per_paper = []
for i, array_str in enumerate(json_arrays, start=1):
    try:
        qas = json.loads(array_str)
        ground_truth_per_paper.append(qas)
    except json.JSONDecodeError as e:
        print(f"Error parsing array {i}: {e}")

print(f"Loaded ground truth for {len(ground_truth_per_paper)} papers.")
import json

output_path = "truth_parsed.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(ground_truth_per_paper, f, ensure_ascii=False, indent=2)

print(f"Saved parsed ground truth to {output_path}")
