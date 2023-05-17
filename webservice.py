import os
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS

import predict
import config
import utility
import pdb

app = Flask(__name__)
CORS(app)
api = Api(app)

model_lexicon = predict.load_keras_model(model_path=config.model_path_lexicon)
model_simple = predict.load_keras_model(model_path=config.model_path)


class PredictLexicon(Resource):
    @staticmethod
    def get():
        comment = request.args.get('text', '')
        sentiment = predict.predict_one_lexicon(model_lexicon, comment)
        return sentiment
api.add_resource(PredictLexicon, '/predict_lexicon/')


class Predict(Resource):
    @staticmethod
    def get():
        try:
            comment = request.args.get('text', '')
            sentiment = predict.predict_one(model_simple, comment)
            return sentiment
        except Exception as e:
            return {str(e)}, 500
api.add_resource(Predict, '/predict/')


class Home(Resource):
    @staticmethod
    def get():
        root = os.getcwd()
        result = utility.get_tree(root)
        return result
api.add_resource(Home, '/')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9001, debug=False, threaded=False)

