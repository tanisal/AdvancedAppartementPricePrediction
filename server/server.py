from flask import Flask,request,jsonify
from flask_cors import CORS
import util


app= Flask(__name__)
CORS(app)

@app.route('/get_location_names')
def get_location_names():
    response = jsonify({
        'locations':util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/get_build_method')
def get_build_method():
    response = jsonify({
        'build':util.get_build_method()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_appartament_price', methods=['POST'])
def predict_appartament_price():
    total_square = float(request.form['total_square'])
    location = request.form['location']
    floor = int(request.form['floor'])
    rooms =int(request.form['rooms'])
    build = request.form['build']
    
    response = jsonify({
        'estimated_price':util.get_predict_price(location,total_square,rooms,floor,build)

    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response



if __name__ == '__main__':
    print('Starting Python Flask Server for Real Estate Price Prediction...')
    util.load_saved_artifacts()
    app.run()

