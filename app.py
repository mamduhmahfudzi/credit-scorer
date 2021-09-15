from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from predict import make_predictions

def create_app():
    """ app factories """
    app = Flask(__name__)
    CORS(app)


    @app.route("/", methods=["GET"])
    def default():
        return render_template("index.html")


    @app.route('/predict/', methods=['GET'])
    def predict():
        if request.method == 'GET':
            data = {}
            data["Age"] = request.args.get("Age")
            data["Sex"] = request.args.get("Sex")
            data["Job"] = request.args.get("Job")
            data["Housing"] = request.args.get("Housing")
            data["Saving accounts"] = request.args.get("saving_account")
            data["Checking account"] = request.args.get("checking_account")
            data["Credit amount"] = request.args.get("credit_amount")
            data["Duration"] = request.args.get("duration")
            data["Purpose"] = request.args.get("purpose")
            result = make_predictions(data)
            result = {
                "model_version": "german_credit_1.0.0",
                "api_version": "v1",
                "result": str(round(list(result)[0], 3))
            }
            print(result)
        return jsonify(result)
    return app
