from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/api/numbers', methods=["POST"])
def load_numbers():
    data = request.get_json()

    try:
        nums = [int(data[f'num{i}']) for i in range(1, 5)]
        for num in nums:
            if num < 0:
                raise ValueError("invalid entity")
    except (ValueError, KeyError):
        return jsonify({'error': 'Invalid input'}), 422

    result = sum(nums)
    return jsonify({'sum': result}), 200

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)