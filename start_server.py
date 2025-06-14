# start_server.py
from pyngrok import ngrok
import subprocess

# Start ngrok tunnel on port 8000
public_url = ngrok.connect(8000)
print("ngrok tunnel URL:", public_url)

# Start uvicorn using subprocess
subprocess.run(["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"])
