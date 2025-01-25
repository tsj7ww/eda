from flask import Flask, jsonify
import kfp

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/pipelines')
def list_pipelines():
    client = kfp.Client()
    pipelines = client.list_pipelines()
    return jsonify({"pipelines": pipelines})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

# kubeflow/pipelines/api_server.py
from flask import Flask, request, jsonify
from kfp_server_api import ApiClient
import os

app = Flask(__name__)
api_client = ApiClient()

@app.route('/api/v1/pipelines', methods=['GET'])
def list_pipelines():
    return jsonify({"pipelines": []})

@app.route('/api/v1/runs', methods=['POST'])
def create_run():
    return jsonify({"run_id": "sample-run-id"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)