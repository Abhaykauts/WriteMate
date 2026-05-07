import threading
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
import recognition as rec

app = Flask(__name__)
CORS(app)

# Global state to share data between threads
state = {
    "buffer": "",
    "is_running": False,
    "mode": None, # 'voice' or 'gesture'
    "online": False
}

def run_recognition():
    global state
    state["is_running"] = True
    state["buffer"] = ""
    
    if state["mode"] == "voice":
        rec.voice_mode_backend(state)
    elif state["mode"] == "gesture":
        rec.gesture_mode_backend(state)
    
    state["is_running"] = False

@app.route('/start/<mode>', methods=['GET'])
@app.route('/start/<mode>/<online>', methods=['GET'])
def start(mode, online="false"):
    if state["is_running"]:
        return jsonify({"status": "already_running"}), 400
    
    if mode not in ["voice", "gesture"]:
        return jsonify({"status": "invalid_mode"}), 400
    
    state["mode"] = mode
    state["online"] = (online.lower() == "true")
    threading.Thread(target=run_recognition, daemon=True).start()
    return jsonify({"status": "started", "mode": mode, "online": state["online"]})

@app.route('/stop', methods=['GET'])
def stop():
    state["is_running"] = False
    return jsonify({"status": "stopping", "final_text": state["buffer"]})

@app.route('/get-text', methods=['GET'])
def get_text():
    return jsonify({
        "text": state["buffer"], 
        "is_running": state["is_running"]
    })

if __name__ == '__main__':
    print("WriteMate Backend running on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
