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