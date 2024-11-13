from flask import Flask, request, jsonify
import os
import requests
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()
# //TODO remove print after test
print("OpenAI Key:", os.getenv('OPENAI_API_KEY'))
print("Anthropic Key:", os.getenv('ANTHROPIC_API_KEY'))

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    model = data.get('model')
    system_prompt = data.get('systemPrompt')
    user_data = data.get('userData')

    if not model or not system_prompt or not user_data:
        missing_fields = []
        if not model:
            missing_fields.append("model")
        if not system_prompt:
            missing_fields.append("systemPrompt")
        if not user_data:
            missing_fields.append("userData")
        
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

    try:
        if model == 'openai':
            response_text = interact_with_openai(system_prompt, user_data)
        elif model == 'anthropic':
            response_text = interact_with_anthropic(system_prompt, user_data)
        else:
            return jsonify({"error": "Invalid model selection. Choose 'openai' or 'anthropic'."}), 400
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Network error occurred: {str(e)}"}), 503
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

    return jsonify({"response": response_text})

def interact_with_openai(system_prompt, user_data):
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
    }
    
    payload = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_data}
        ],
        'max_tokens': 150,
    }
    
    try:
        response = requests.post('https://api.openai.com/v1/chat/completions', json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        raise Exception(f"OpenAI API error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        raise Exception(f"Network error occurred while connecting to OpenAI API: {req_err}")
    
    if response.status_code == 200:
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', 'No response available')
    else:
        raise Exception(f"OpenAI API request failed with status code {response.status_code}: {response.text}")

def interact_with_anthropic(system_prompt, user_data):
    headers = {
        'Authorization': f'Bearer {ANTHROPIC_API_KEY}',
    }
    payload = {
        'model': 'claude-3', 
        'prompt': f'{system_prompt}\n\n{user_data}',
        'max_tokens_to_sample': 150,
    }
    
    try:
        response = requests.post('https://api.anthropic.com/v1/complete', json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        raise Exception(f"Anthropic API error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        raise Exception(f"Network error occurred while connecting to Anthropic API: {req_err}")
    
    if response.status_code == 200:
        return response.json().get('completion', 'No response available')
    else:
        raise Exception(f"Anthropic API request failed with status code {response.status_code}: {response.text}")

if __name__ == '__main__':
    app.run(debug=True)
