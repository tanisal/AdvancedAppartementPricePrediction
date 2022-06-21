from flask import Flask, request, jsonify
from flask_cors import CORS
import util

# Cerating a flask server
app = Flask(__name__)

# Handling the cross origin resourse sharing, not to deal with problems with headers
CORS(app)


# -------------------Shumen-------------------------------
# Pass the required root to the decorator
@app.route("/get_location_names_shumen", methods=["GET"])
def get_location_names_shumen():
    """A function that returns the location names from the data columns we have gathered in the main model python file and afterwards were saved in json format"""

    response = jsonify({"locations": util.get_location_names_shumen()})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# Pass the required root to the decorator
@app.route("/get_build_method_shumen")
def get_build_method_shumen():
    """A function that returns the build method gathered, from the data columns gathered in the main moel python file, saved in json format"""
    response = jsonify({"build": util.get_build_method_shumen()})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# Pass the required root to the decorator
@app.route("/predict_appartament_price_shumen", methods=["POST"])
def predict_appartament_price_shumen():
    """A function that requests the data from the form inputed from user  on the http call and uses the data as arguments in a function predicting the price and returns the price"""
    total_square = float(request.form["total_square"])
    location = request.form["location"]
    floor = int(request.form["floor"])
    rooms = int(request.form["rooms"])
    build = request.form["build"]

    response = jsonify(
        {
            "estimated_price": util.get_predict_price_shumen(
                location, total_square, rooms, floor, build
            )
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# --------------------------Plovdiv-----------------------------------------
# Pass the required root to the decorator
@app.route("/get_location_names_plovdiv", methods=["GET"])
def get_location_names_plovdiv():
    """A function that returns the location names from the data columns we have gathered in the main model python file and afterwards were saved in json format"""

    response = jsonify({"locations": util.get_location_names_plovdiv()})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# Pass the required root to the decorator
@app.route("/get_build_method_plovdiv", methods=["GET"])
def get_build_method_plovdiv():
    """A function that returns the build method gathered, from the data columns gathered in the main moel python file, saved in json format"""

    response = jsonify({"build": util.get_build_method_plovdiv()})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# Pass the required root to the decorator
@app.route("/predict_appartament_price_plovdiv", methods=["POST"])
def predict_appartament_price_plovdiv():
    """A function that requests the data from the form inputed from user  on the http call and uses the data as arguments in a function predicting the price and returns the price"""

    total_square = float(request.form["total_square"])
    location = request.form["location"]
    floor = int(request.form["floor"])
    rooms = int(request.form["rooms"])
    build = request.form["build"]

    response = jsonify(
        {
            "estimated_price": util.get_predict_price_plovdiv(
                location, total_square, rooms, floor, build
            )
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# -------------------------Veliko Tarnovo-----------------
# Pass the required root to the decorator
@app.route("/get_location_names_tarnovo")
def get_location_names_tarnovo():
    """A function that returns the location names from the data columns we have gathered in the main model python file and afterwards were saved in json format"""
    response = jsonify({"locations": util.get_location_names_tarnovo()})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# Pass the required root to the decorator
@app.route("/get_build_method_tarnovo")
def get_build_method_tarnovo():
    """A function that returns the build method gathered, from the data columns gathered in the main moel python file, saved in json format"""

    response = jsonify({"build": util.get_build_method_tarnovo()})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# Pass the required root to the decorator
@app.route("/predict_appartament_price_tarnovo", methods=["POST"])
def predict_appartament_price_tarnovo():
    """A function that requests the data from the form inputed from user  on the http call and uses the data as arguments in a function predicting the price and returns the price"""
    total_square = float(request.form["total_square"])
    location = request.form["location"]
    floor = int(request.form["floor"])
    rooms = int(request.form["rooms"])
    build = request.form["build"]

    response = jsonify(
        {
            "estimated_price": util.get_predict_price_tarnovo(
                location, total_square, rooms, floor, build
            )
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# --------------------------Varna-------------------------------

# Pass the required root to the decorator
@app.route("/get_location_names_varna")
def get_location_names_varna():
    """A function that returns the location names from the data columns we have gathered in the main model python file and afterwards were saved in json format"""
    response = jsonify({"locations": util.get_location_names_varna()})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# Pass the required root to the decorator
@app.route("/get_build_method_varna")
def get_build_method_varna():
    """A function that returns the build method gathered, from the data columns gathered in the main moel python file, saved in json format"""
    response = jsonify({"build": util.get_build_method_varna()})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# Pass the required root to the decorator
@app.route("/predict_appartament_price_varna", methods=["POST"])
def predict_appartament_price_varna():
    """A function that requests the data from the form inputed from user  on the http call and uses the data as arguments in a function predicting the price and returns the price"""
    total_square = float(request.form["total_square"])
    location = request.form["location"]
    floor = int(request.form["floor"])
    rooms = int(request.form["rooms"])
    build = request.form["build"]

    response = jsonify(
        {
            "estimated_price": util.get_predict_price_varna(
                location, total_square, rooms, floor, build
            )
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# -----------------------Sofia----------------------------------------

# Pass the required root to the decorator
@app.route("/get_location_names_sofia")
def get_location_names_sofia():
    """A function that returns the location names from the data columns we have gathered in the main model python file and afterwards were saved in json format"""
    response = jsonify({"locations": util.get_location_names_sofia()})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# Pass the required root to the decorator
@app.route("/get_build_method_sofia")
def get_build_method_sofia():
    """A function that returns the build method gathered, from the data columns gathered in the main moel python file, saved in json format"""
    response = jsonify({"build": util.get_build_method_sofia()})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# Pass the required root to the decorator
@app.route("/predict_appartament_price_sofia", methods=["POST"])
def predict_appartament_price_sofia():
    """A function that requests the data from the form inputed from user  on the http call and uses the data as arguments in a function predicting the price and returns the price"""

    total_square = float(request.form["total_square"])
    location = request.form["location"]
    floor = int(request.form["floor"])
    rooms = int(request.form["rooms"])
    build = request.form["build"]

    response = jsonify(
        {
            "estimated_price": util.get_predict_price_sofia(
                location, total_square, rooms, floor, build
            )
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    print("Starting Python Flask Server for Real Estate Price Prediction...")
    # Loading the data for the different cities
    util.load_saved_artifacts_shumen()
    util.load_saved_artifacts_plovdiv()
    util.load_saved_artifacts_tarnovo()
    util.load_saved_artifacts_varna()
    util.load_saved_artifacts_sofia()

    # Run the flask server
    app.run()
