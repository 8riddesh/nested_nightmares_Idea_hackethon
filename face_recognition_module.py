import subprocess
import sys

from flask import jsonify, app


@app.route('/start-voice', methods=['GET'])
def start_voice():
    print(f"Python executable path: {sys.executable}")  # Debugging
    subprocess.Popen(["python", "voice_gui.py"])
    return jsonify({"status": "success", "message": "Voice GUI started!"})