from litellm import completion
import json 

## GET CREDENTIALS 
## RUN ## 
# !gcloud auth application-default login - run this to add vertex credentials to your env
## OR ## 
file_path = 'src/pdf_extractor/vertex_ai_service_account.json'

# Load the JSON file
with open(file_path, 'r') as file:
    vertex_credentials = json.load(file)

# Convert to JSON string
vertex_credentials_json = json.dumps(vertex_credentials)


## COMPLETION CALL 
response = completion(
  model="vertex_ai/gemini-2.5-pro-exp-03-25",
  messages=[{ "content": "What is the capital of France","role": "user"}],
  vertex_credentials=vertex_credentials_json,
  vertex_ai_location="us-central1"  # This parameter is documented
)


result = response.choices[0].message.content
print(result)