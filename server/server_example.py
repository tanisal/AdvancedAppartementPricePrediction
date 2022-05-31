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