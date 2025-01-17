from flask import Flask, request, jsonify, render_template_string
from inference import load_model, generate_text
import torch

app = Flask(__name__)

try:
    model, sp, device = load_model()
except Exception as e:
    print(f"Error loading model: {e}")
    raise


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Poem GPT</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        textarea { width: 100%; height: 100px; margin: 10px 0; }
        button { padding: 10px 20px; }
        #result { margin-top: 20px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>Poem GPT</h1>
    <form id="generateForm">
        <textarea id="prompt" placeholder="Başlangıç metni girin..."></textarea><br>
        <button type="submit">Metin Üret</button>
    </form>
    <div id="result"></div>

    <script>
        document.getElementById('generateForm').onsubmit = async (e) => {
            e.preventDefault();
            const prompt = document.getElementById('prompt').value;
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt, max_tokens: 300})
            });
            const data = await response.json();
            document.getElementById('result').textContent = data.generated_text;
        };
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 100)
        
        generated_text = generate_text(model, sp, prompt, max_tokens, device)
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)