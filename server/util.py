import json
import numpy as np
from scipy.special import inv_boxcox1p
from joblib import load

# -------------Utility Part-----------


# -----Shumen-------

# Defining global variables for Shumen
__locations_shumen = None
__data_columns_shumen = None
__build_method_shumen = None
__elasticnet_shumen = None
__ridge_shumen = None
__lasso_shumen = None
__gbr_shumen = None
__xgboost_shumen = None
__stack_shumen = None


# Blending the differrent models
def blend_model_shumen(X):
    return (
        (0.05 * __elasticnet_shumen.predict(X))
        + (0.1 * __lasso_shumen.predict(X))
        + (0.05 * __ridge_shumen.predict(X))
        + (0.2 * __gbr_shumen.predict(X))
        + (0.3 * __xgboost_shumen.predict(np.array(X)))
        + (0.3 * __stack_shumen.predict(np.array(X)))
    )


def get_predict_price_shumen(location, m2, rooms, floor, build):
    """A function that returns the predicted price, using the regresssion model and user inputs"""
    try:
        # finding the index of the filled in location, converted to lower case
        loc_index = __data_columns_shumen.index(location.lower())
    except:
        loc_index = -1
    try:
        # similar for building method
        build_index = __data_columns_shumen.index(build.lower())
    except:
        build_index = -1

    # Creating an array long as the lenght of number of columns
    x = np.zeros(len(__data_columns_shumen))
    # Assigning m2, room and floor inputs
    x[0], x[1], x[2] = m2, rooms, floor
    # Using the location index to assign a dummy
    if loc_index >= 0:
        x[loc_index] = 1
    # Same for the build method
    if build_index >= 0:
        x[build_index] = 1

    # We use the regressions' model predict function
    return round(blend_model_shumen([x])[0], 0)


def get_location_names_shumen():
    """Function which returns locations data"""

    return __locations_shumen


def get_build_method_shumen():
    """Function which returns building method data"""

    return __build_method_shumen


def load_saved_artifacts_shumen():
    """Function which loads server artifacts files, from which we extract data , locations and build method and model, used for prediction"""

    print("loading saved Shumen artifacts...start")
    global __data_columns_shumen
    global __locations_shumen
    global __build_method_shumen
    global __elasticnet_shumen
    global __ridge_shumen
    global __lasso_shumen
    global __gbr_shumen
    global __xgboost_shumen
    global __stack_shumen

    # We load the columns data from the json file
    with open("./artifacts/shumen_columns.json", "r", encoding="utf-8") as f:
        # loading all the columns
        __data_columns_shumen = json.load(f)["data_columns"]
        # here we define just the location columns
        __locations_shumen = __data_columns_shumen[3:-3]
        # the same for the build method
        __build_method_shumen = __data_columns_shumen[-3:]

    # We load the model from the joblib/pickle file
    __elasticnet_shumen = load(
        "./artifacts/shumen_appartament_price_model_elasticnet.joblib"
    )
    __ridge_shumen = load("./artifacts/shumen_appartament_price_model_ridge.joblib")
    __lasso_shumen = load("./artifacts/shumen_appartament_price_model_lasso.joblib")
    __gbr_shumen = load("./artifacts/shumen_appartament_price_model_gbr.joblib")
    __xgboost_shumen = load("./artifacts/shumen_appartament_price_model_xgboost.joblib")
    __stack_shumen = load("./artifacts/shumen_appartament_price_model_stack.joblib")

    print("loading saved  Shumen artifacts...done")


# # --------------------------------Plovdiv----------------------

# Defining global variables for Plovdiv
__locations_plovdiv = None
__data_columns_plovdiv = None
__build_method_plovdiv = None
__elasticnet_plovdiv = None
__ridge_plovdiv = None
__lasso_plovdiv = None
__gbr_plovdiv = None
__xgboost_plovdiv = None
__stack_plovdiv = None


# Blending the differrent models
def blend_model_plovdiv(X):
    return (
        (0.05 * __elasticnet_plovdiv.predict(X))
        + (0.1 * __lasso_plovdiv.predict(X))
        + (0.05 * __ridge_plovdiv.predict(X))
        + (0.2 * __gbr_plovdiv.predict(X))
        + (0.3 * __xgboost_plovdiv.predict(np.array(X)))
        + (0.3 * __stack_plovdiv.predict(np.array(X)))
    )


def get_predict_price_plovdiv(location, m2, rooms, floor, build):
    """A function that returns the predicted price, using the regresssion model and user inputs"""
    try:
        # finding the index of the filled in location, converted to lower case
        loc_index = __data_columns_plovdiv.index(location.lower())
    except:
        loc_index = -1
    try:
        # similar for building method
        build_index = __data_columns_plovdiv.index(build.lower())
    except:
        build_index = -1

    # Creating an array long as the lenght of number of columns
    x = np.zeros(len(__data_columns_plovdiv))
    # Assigning m2, room and floor inputs
    x[0], x[1], x[2] = m2, rooms, floor
    # Using the location index to assign a dummy
    if loc_index >= 0:
        x[loc_index] = 1
    # Same for the build method
    if build_index >= 0:
        x[build_index] = 1

    # We use the regressions' model predict function
    return round(inv_boxcox1p(blend_model_plovdiv([x])[0], 0.3), 0)


def get_location_names_plovdiv():
    """Function which returns locations data"""

    return __locations_plovdiv


def get_build_method_plovdiv():
    """Function which returns building method data"""

    return __build_method_plovdiv


def load_saved_artifacts_plovdiv():
    """Function which loads server artifacts files, from which we extract data , locations and build method and model, used for prediction"""

    print("loading saved Plovdiv artifacts...start")
    global __data_columns_plovdiv
    global __locations_plovdiv
    global __build_method_plovdiv
    global __elasticnet_plovdiv
    global __ridge_plovdiv
    global __lasso_plovdiv
    global __gbr_plovdiv
    global __xgboost_plovdiv
    global __stack_plovdiv

    # We load the columns data from the json file
    with open("./artifacts/plovdiv_columns.json", "r", encoding="utf-8") as f:
        # loading all the columns
        __data_columns_plovdiv = json.load(f)["data_columns"]
        # here we define just the location columns
        __locations_plovdiv = __data_columns_plovdiv[3:-3]
        # the same for the build method
        __build_method_plovdiv = __data_columns_plovdiv[-3:]

    # We load the model from the joblib/pickle file
    __elasticnet_plovdiv = load(
        "./artifacts/plovdiv_appartament_price_model_elasticnet.joblib"
    )
    __ridge_plovdiv = load("./artifacts/plovdiv_appartament_price_model_ridge.joblib")
    __lasso_plovdiv = load("./artifacts/plovdiv_appartament_price_model_lasso.joblib")
    __gbr_plovdiv = load("./artifacts/plovdiv_appartament_price_model_gbr.joblib")
    __xgboost_plovdiv = load(
        "./artifacts/plovdiv_appartament_price_model_xgboost.joblib"
    )
    __stack_plovdiv = load("./artifacts/plovdiv_appartament_price_model_stack.joblib")

    print("loading saved  Plovdiv artifacts...done")


# # -----------------------------Veliko Tarnovo--------------

# Defining global variables for Tarnovo
__locations_tarnovo = None
__data_columns_tarnovo = None
__build_method_tarnovo = None
__elasticnet_tarnovo = None
__ridge_tarnovo = None
__lasso_tarnovo = None
__gbr_tarnovo = None
__xgboost_tarnovo = None
__stack_tarnovo = None


# Blending the differrent models
def blend_model_tarnovo(X):
    return (
        (0.05 * __elasticnet_tarnovo.predict(X))
        + (0.1 * __lasso_tarnovo.predict(X))
        + (0.05 * __ridge_tarnovo.predict(X))
        + (0.2 * __gbr_tarnovo.predict(X))
        + (0.3 * __xgboost_tarnovo.predict(np.array(X)))
        + (0.3 * __stack_tarnovo.predict(np.array(X)))
    )


def get_predict_price_tarnovo(location, m2, rooms, floor, build):
    """A function that returns the predicted price, using the regresssion model and user inputs"""
    try:
        # finding the index of the filled in location, converted to lower case
        loc_index = __data_columns_tarnovo.index(location.lower())
    except:
        loc_index = -1
    try:
        # similar for building method
        build_index = __data_columns_tarnovo.index(build.lower())
    except:
        build_index = -1

    # Creating an array long as the lenght of number of columns
    x = np.zeros(len(__data_columns_tarnovo))
    # Assigning m2, room and floor inputs
    x[0], x[1], x[2] = m2, rooms, floor
    # Using the location index to assign a dummy
    if loc_index >= 0:
        x[loc_index] = 1
    # Same for the build method
    if build_index >= 0:
        x[build_index] = 1

    # We use the regressions' model predict function
    return round(inv_boxcox1p(blend_model_tarnovo([x])[0], 0.35), 0)


def get_location_names_tarnovo():
    """Function which returns locations data"""

    return __locations_tarnovo


def get_build_method_tarnovo():
    """Function which returns building method data"""

    return __build_method_tarnovo


def load_saved_artifacts_tarnovo():
    """Function which loads server artifacts files, from which we extract data , locations and build method and model, used for prediction"""

    print("loading saved Tarnovo artifacts...start")
    global __data_columns_tarnovo
    global __locations_tarnovo
    global __build_method_tarnovo
    global __elasticnet_tarnovo
    global __ridge_tarnovo
    global __lasso_tarnovo
    global __gbr_tarnovo
    global __xgboost_tarnovo
    global __stack_tarnovo

    # We load the columns data from the json file
    with open("./artifacts/tarnovo_columns.json", "r", encoding="utf-8") as f:
        # loading all the columns
        __data_columns_tarnovo = json.load(f)["data_columns"]
        # here we define just the location columns
        __locations_tarnovo = __data_columns_tarnovo[3:-3]
        # the same for the build method
        __build_method_tarnovo = __data_columns_tarnovo[-3:]

    # We load the model from the joblib/pickle file
    __elasticnet_tarnovo = load(
        "./artifacts/tarnovo_appartament_price_model_elasticnet.joblib"
    )
    __ridge_tarnovo = load("./artifacts/tarnovo_appartament_price_model_ridge.joblib")
    __lasso_tarnovo = load("./artifacts/tarnovo_appartament_price_model_lasso.joblib")
    __gbr_tarnovo = load("./artifacts/tarnovo_appartament_price_model_gbr.joblib")
    __xgboost_tarnovo = load(
        "./artifacts/tarnovo_appartament_price_model_xgboost.joblib"
    )
    __stack_tarnovo = load("./artifacts/tarnovo_appartament_price_model_stack.joblib")

    print("loading saved  Tarnovo artifacts...done")


# # ------------------------------------Varna------------------------------------

# Defining global variables for Varna
__locations_varna = None
__data_columns_varna = None
__build_method_varna = None
__elasticnet_varna = None
__ridge_varna = None
__lasso_varna = None
__gbr_varna = None
__xgboost_varna = None
__stack_varna = None


# Blending the differrent models
def blend_model_varna(X):
    return (
        (0.05 * __elasticnet_varna.predict(X))
        + (0.1 * __lasso_varna.predict(X))
        + (0.05 * __ridge_varna.predict(X))
        + (0.2 * __gbr_varna.predict(X))
        + (0.3 * __xgboost_varna.predict(np.array(X)))
        + (0.3 * __stack_varna.predict(np.array(X)))
    )


def get_predict_price_varna(location, m2, rooms, floor, build):
    """A function that returns the predicted price, using the regresssion model and user inputs"""
    try:
        # finding the index of the filled in location, converted to lower case
        loc_index = __data_columns_varna.index(location.lower())
    except:
        loc_index = -1
    try:
        # similar for building method
        build_index = __data_columns_varna.index(build.lower())
    except:
        build_index = -1

    # Creating an array long as the lenght of number of columns
    x = np.zeros(len(__data_columns_varna))
    # Assigning m2, room and floor inputs
    x[0], x[1], x[2] = m2, rooms, floor
    # Using the location index to assign a dummy
    if loc_index >= 0:
        x[loc_index] = 1
    # Same for the build method
    if build_index >= 0:
        x[build_index] = 1

    # We use the regressions' model predict function
    return round(inv_boxcox1p(blend_model_varna([x])[0], -0.22), 0)


def get_location_names_varna():
    """Function which returns locations data"""

    return __locations_varna


def get_build_method_varna():
    """Function which returns building method data"""

    return __build_method_varna


def load_saved_artifacts_varna():
    """Function which loads server artifacts files, from which we extract data , locations and build method and model, used for prediction"""

    print("loading saved Varna artifacts...start")
    global __data_columns_varna
    global __locations_varna
    global __build_method_varna
    global __elasticnet_varna
    global __ridge_varna
    global __lasso_varna
    global __gbr_varna
    global __xgboost_varna
    global __stack_varna

    # We load the columns data from the json file
    with open("./artifacts/varna_columns.json", "r", encoding="utf-8") as f:
        # loading all the columns
        __data_columns_varna = json.load(f)["data_columns"]
        # here we define just the location columns
        __locations_varna = __data_columns_varna[3:-3]
        # the same for the build method
        __build_method_varna = __data_columns_varna[-3:]

    # We load the model from the joblib/pickle file
    __elasticnet_varna = load(
        "./artifacts/varna_appartament_price_model_elasticnet.joblib"
    )
    __ridge_varna = load("./artifacts/varna_appartament_price_model_ridge.joblib")
    __lasso_varna = load("./artifacts/varna_appartament_price_model_lasso.joblib")
    __gbr_varna = load("./artifacts/varna_appartament_price_model_gbr.joblib")
    __xgboost_varna = load("./artifacts/varna_appartament_price_model_xgboost.joblib")
    __stack_varna = load("./artifacts/varna_appartament_price_model_stack.joblib")

    print("loading saved Varna artifacts...done")


# # ----------------------------------------- Sofia-------------

# Defining global variables for Sofia
__locations_sofia = None
__data_columns_sofia = None
__build_method_sofia = None
__elasticnet_sofia = None
__ridge_sofia = None
__lasso_sofia = None
__gbr_sofia = None
__xgboost_sofia = None
__stack_sofia = None


# Blending the differrent models
def blend_model_sofia(X):
    return (
        (0.05 * __elasticnet_sofia.predict(X))
        + (0.1 * __lasso_sofia.predict(X))
        + (0.05 * __ridge_sofia.predict(X))
        + (0.2 * __gbr_sofia.predict(X))
        + (0.3 * __xgboost_sofia.predict(np.array(X)))
        + (0.3 * __stack_sofia.predict(np.array(X)))
    )


def get_predict_price_sofia(location, m2, rooms, floor, build):
    """A function that returns the predicted price, using the regresssion model and user inputs"""
    try:
        # finding the index of the filled in location, converted to lower case
        loc_index = __data_columns_sofia.index(location.lower())
    except:
        loc_index = -1
    try:
        # similar for building method
        build_index = __data_columns_sofia.index(build.lower())
    except:
        build_index = -1

    # Creating an array long as the lenght of number of columns
    x = np.zeros(len(__data_columns_sofia))
    # Assigning m2, room and floor inputs
    x[0], x[1], x[2] = m2, rooms, floor
    # Using the location index to assign a dummy
    if loc_index >= 0:
        x[loc_index] = 1
    # Same for the build method
    if build_index >= 0:
        x[build_index] = 1

    # We use the regressions' model predict function
    return round(inv_boxcox1p(blend_model_sofia([x])[0], 0.07), 0)


def get_location_names_sofia():
    """Function which returns locations data"""

    return __locations_sofia


def get_build_method_sofia():
    """Function which returns building method data"""

    return __build_method_sofia


def load_saved_artifacts_sofia():
    """Function which loads server artifacts files, from which we extract data , locations and build method and model, used for prediction"""

    print("loading saved Sofia artifacts...start")
    global __data_columns_sofia
    global __locations_sofia
    global __build_method_sofia
    global __elasticnet_sofia
    global __ridge_sofia
    global __lasso_sofia
    global __gbr_sofia
    global __xgboost_sofia
    global __stack_sofia

    # We load the columns data from the json file
    with open("./artifacts/sofia_columns.json", "r", encoding="utf-8") as f:
        # loading all the columns
        __data_columns_sofia = json.load(f)["data_columns"]
        # here we define just the location columns
        __locations_sofia = __data_columns_sofia[3:-3]
        # the same for the build method
        __build_method_sofia = __data_columns_sofia[-3:]

    # We load the model from the joblib/pickle file
    __elasticnet_sofia = load(
        "./artifacts/sofia_appartament_price_model_elasticnet.joblib"
    )
    __ridge_sofia = load("./artifacts/sofia_appartament_price_model_ridge.joblib")
    __lasso_sofia = load("./artifacts/sofia_appartament_price_model_lasso.joblib")
    __gbr_sofia = load("./artifacts/sofia_appartament_price_model_gbr.joblib")
    __xgboost_sofia = load("./artifacts/sofia_appartament_price_model_xgboost.joblib")
    __stack_sofia = load("./artifacts/sofia_appartament_price_model_stack.joblib")

    print("loading saved  Sofia artifacts...done")


# Loading all the artifacts data for the different cities, also checking if prediction functions works fine
if __name__ == "__main__":
    print("Starting Python Flask Server for Real Estate Price Prediction...")
    load_saved_artifacts_shumen()
    print(get_location_names_shumen())
    print(get_build_method_shumen())
    print(get_predict_price_shumen("тракия", 58, 3, 9, "епк"))

    load_saved_artifacts_plovdiv()
    print(get_location_names_plovdiv())
    print(get_build_method_plovdiv())
    print(get_predict_price_plovdiv("център", 58, 3, 9, "тухла"))

    load_saved_artifacts_tarnovo()
    print(get_location_names_tarnovo())
    print(get_build_method_tarnovo())
    print(get_predict_price_tarnovo("център", 58, 3, 9, "тухла"))

    load_saved_artifacts_varna()
    print(get_location_names_varna())
    print(get_build_method_varna())
    print(get_predict_price_varna("център", 58, 3, 9, "тухла"))

    load_saved_artifacts_sofia()
    print(get_location_names_sofia())
    print(get_build_method_sofia())
    print(get_predict_price_sofia("лозенец", 80, 3, 2, "епк"))
