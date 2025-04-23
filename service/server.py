from flask import Flask, request, jsonify, render_template
from logging.config import dictConfig
import config as cfg

dictConfig(cfg.log_config)

app = Flask(__name__)

@app.route('/api/numbers', methods=["POST"])
def process_numbers():
    request_body = request.get_json()
    app.logger.info(f"handling \"process_numbers\": request body: {request_body}")
    try:
        nums = [int(request_body[f'num{i}']) for i in range(1, 5)]
        for num in nums:
            if num < 0:
                raise ValueError("invalid entity")
    except (ValueError, KeyError):
        app.logger.warning("Received invalid request body: %s", request_body)
        return jsonify({'error': 'Invalid input'}), 422

    result = sum(nums)
    app.logger.info(f"client response \"process_numbers\": {result}")
    return jsonify({'sum': result}), 200

@app.route("/")
def home():
    app.logger.debug("server started")
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)