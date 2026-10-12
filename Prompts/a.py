import json 
path = r"C:\Users\Krish\Downloads\LangChain\Prompts\prompt_templates\template_1.json"
with open(path,'r') as f:
        prompt = json.load(f)

print(prompt)