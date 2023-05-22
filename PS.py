import openai
import json

# Set up your OpenAI API credentials
openai.api_key = 'YOUR_API_KEY'

# Define a function to generate a response from ChatGPT
def generate_response(question):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=question,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# List of questions
questions = [
    "What are the symptoms of the common cold?",
    "How can I prevent the flu?",
    "What should I do if I have a persistent cough?",
    # Add more questions here
]

# Generate responses and save in JSON format
responses = {}
for question in questions:
    answer = generate_response(question)
    responses[question] = answer

# Save responses in JSON file
with open('responses.json', 'w') as json_file:
    json.dump(responses, json_file, indent=4)

print("Responses saved successfully in JSON format.")
