from flask import Flask,request,jsonify
from flask_cors import CORS
import util


app= Flask(__name__)
CORS(app)


#-------------------Shumen-------------------------------
@app.route('/get_location_names_shumen')
def get_location_names_shumen():
    response = jsonify({
        'locations':util.get_location_names_shumen()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/get_build_method_shumen')
def get_build_method_shumen():
    response = jsonify({
        'build':util.get_build_method_shumen()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_appartament_price_shumen', methods=['POST'])
def predict_appartament_price_shumen():
    total_square = float(request.form['total_square'])
    location = request.form['location']
    floor = int(request.form['floor'])
    rooms =int(request.form['rooms'])
    build = request.form['build']
    
    response = jsonify({
        'estimated_price':util.get_predict_price_shumen(location,total_square,rooms,floor,build)

    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response



#--------------------------Plovdiv-----------------------------------------

@app.route('/get_location_names_plovdiv')
def get_location_names_plovdiv():
    response = jsonify({
        'locations':util.get_location_names_plovdiv()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/get_build_method_plovdiv')
def get_build_method_plovdiv():
    response = jsonify({
        'build':util.get_build_method_plovdiv()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_appartament_price_plovdiv', methods=['POST'])
def predict_appartament_price_plovdiv():
    total_square = float(request.form['total_square'])
    location = request.form['location']
    floor = int(request.form['floor'])
    rooms =int(request.form['rooms'])
    build = request.form['build']
    
    response = jsonify({
        'estimated_price':util.get_predict_price_plovdiv(location,total_square,rooms,floor,build)

    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response



#-------------------------Veliko Tarnovo-----------------

@app.route('/get_location_names_tarnovo')
def get_location_names_tarnovo():
    response = jsonify({
        'locations':util.get_location_names_tarnovo()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/get_build_method_tarnovo')
def get_build_method_tarnovo():
    response = jsonify({
        'build':util.get_build_method_tarnovo()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_appartament_price_tarnovo', methods=['POST'])
def predict_appartament_price_tarnovo():
    total_square = float(request.form['total_square'])
    location = request.form['location']
    floor = int(request.form['floor'])
    rooms =int(request.form['rooms'])
    build = request.form['build']
    
    response = jsonify({
        'estimated_price':util.get_predict_price_tarnovo(location,total_square,rooms,floor,build)

    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


#--------------------------Varna-------------------------------

@app.route('/get_location_names_varna')
def get_location_names_varna():
    response = jsonify({
        'locations':util.get_location_names_varna()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/get_build_method_varna')
def get_build_method_varna():
    response = jsonify({
        'build':util.get_build_method_varna()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_appartament_price_varna', methods=['POST'])
def predict_appartament_price_varna():
    total_square = float(request.form['total_square'])
    location = request.form['location']
    floor = int(request.form['floor'])
    rooms =int(request.form['rooms'])
    build = request.form['build']
    
    response = jsonify({
        'estimated_price':util.get_predict_price_varna(location,total_square,rooms,floor,build)

    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response



#-----------------------Sofia----------------------------------------

@app.route('/get_location_names_sofia')
def get_location_names_sofia():
    response = jsonify({
        'locations':util.get_location_names_sofia()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/get_build_method_sofia')
def get_build_method_sofia():
    response = jsonify({
        'build':util.get_build_method_sofia()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_appartament_price_sofia', methods=['POST'])
def predict_appartament_price_sofia():
    total_square = float(request.form['total_square'])
    location = request.form['location']
    floor = int(request.form['floor'])
    rooms =int(request.form['rooms'])
    build = request.form['build']
    
    response = jsonify({
        'estimated_price':util.get_predict_price_sofia(location,total_square,rooms,floor,build)

    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    print('Starting Python Flask Server for Real Estate Price Prediction...')
    util.load_saved_artifacts_shumen()
    util.load_saved_artifacts_plovdiv()
    util.load_saved_artifacts_tarnovo()
    util.load_saved_artifacts_varna()
    util.load_saved_artifacts_sofia()
    app.run()

