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