from flask import Flask,request,jsonify

import json
import pickle
import numpy as np



app= Flask(__name__)

@app.route('/get_location_names')
def get_location_names():
    response = jsonify({
        'locations':get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin','*')
    return response

@app.route('/get_build_method')
def get_build_method():
    response = jsonify({
        'build':get_build_method()
    })
    response.headers.add('Access-Control-Allow-Origin','*')
    return response


@app.route('/predict_appartament_price', methods=['POST'])
def predict_appartament_price():
    total_square = float(request.form['total_square'])
    location = request.form['location']
    floor = int(request.form['floor'])
    rooms =int(request.form['rooms'])
    build = request.form['build']
    
    response = jsonify({
        'estimated_price':get_predict_price(location,total_square,rooms,floor,build)

    })
    response.headers.add('Access-Control-Allow-Origin','*')
    return response


#-------------Utility Part-----------


__locations = None
__data_columns = None
__build_method = None
__model = None

def get_predict_price(location,m2,rooms,floor,build):
    try:
        loc_index=__data_columns.index(location.lower())
    except:
        loc_index =-1
    try:
        build_index=__data_columns.index(build.lower())
    except:
        build_index = -1
    x=np.zeros(len(__data_columns))
    x[0]=m2
    x[1]=rooms
    x[2]=floor
    if loc_index >= 0:
        x[loc_index] = 1
    if build_index>= 0:
        x[build_index] =1

    return round(__model.predict([x])[0])


#Function which returns locations list
def get_location_names():
    return __locations

#Function which returns building method list
def get_build_method():
    return __build_method

#Function which loads server artifacts files, from which we extract locations and build method and model
def load_saved_artifacts():
    print('loading saved artifacts...start')
    global __data_columns
    global __locations
    global __build_method
    global __model





    with open('./artifacts/columns.json','r',encoding='utf-8') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:-3]
        __build_method = __data_columns[-3:]

    
    with open('./artifacts/shumen_appartament_price_model.pickle','rb') as f:
        __model = pickle.load(f)
    print('loading saved artifacts...done')
        



if __name__ == '__main__':
    print('Starting Python Flask Server for Real Estate Price Prediction...')
    load_saved_artifacts()
    print(get_location_names())
    print(get_build_method())
    print(get_predict_price('тракия',58,3,9,'тухла'))
    app.run()

