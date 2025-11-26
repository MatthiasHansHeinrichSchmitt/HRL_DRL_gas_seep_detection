from flask import Flask, jsonify, send_from_directory
import threading
import os

# Path to static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = Flask(__name__, static_folder=STATIC_DIR)

# shared state
latest_values = [0.0, 0.0, 0.0]

meshcat_url = "http://127.0.0.1:7000"  #default, will be overwritten

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/bars")
def bars():
    return jsonify(latest_values)


def run_server():
    app.run(host="127.0.0.1",port=7001, debug=False, use_reloader=False)


def start_server(HUGIN_meshcat_url):
    global meshcat_url
    meshcat_url =  HUGIN_meshcat_url
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()