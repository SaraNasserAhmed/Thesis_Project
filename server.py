import cgi

import flask
from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def server_welcomePage():
    return "FLASK is UP"


@app.route('/receiveData', methods=['POST'])
def process_data():
    data = request.get_json()
    # Machine learning model takes part here to generate probabilities
    probabilities = {"p": [45],
                     "li": [70, 90]
                     }
    print(data)
    return flask.jsonify({"probabilities": probabilities})




if __name__ == "__main__":
    app.run(debug=True)
