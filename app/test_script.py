import requests

url = "http://localhost:11434/api/chat"

payload = {
    "model": "llama3.2:latest",
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "stream": False
}

response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("Response:", response.text)