__locations_tarnovo = None
__data_columns_tarnovo = None
__build_method_tarnovo = None
__model_tarnovo = None

def get_predict_price_tarnovo(location,m2,rooms,floor,build):
    try:
        loc_index=__data_columns_tarnovo.index(location.lower())
    except:
        loc_index =-1
    try:
        build_index=__data_columns_tarnovo.index(build.lower())
    except:
        build_index = -1
    x=np.zeros(len(__data_columns_tarnovo))
    x[0]=m2
    x[1]=rooms
    x[2]=floor
    if loc_index >= 0:
        x[loc_index] = 1
    if build_index>= 0:
        x[build_index] =1

    return round(__model_tarnovo.predict([x])[0])


#Function which returns locations list
def get_location_names_tarnovo():
    return __locations_tarnovo

#Function which returns building method list
def get_build_method_tarnovo():
    return __build_method_tarnovo

#Function which loads server artifacts files, from which we extract locations and build method and model
def load_saved_artifacts_tarnovo():
    print('loading saved Tarnovo artifacts...start')
    global __data_columns_tarnovo
    global __locations_tarnovo
    global __build_method_tarnovo
    global __model_tarnovo

    with open('./artifacts/tarnovo_columns.json','r',encoding='utf-8') as f:
        __data_columns_tarnovo = json.load(f)['data_columns']
        __locations_tarnovo = __data_columns_tarnovo[3:-3]
        __build_method_tarnovo = __data_columns_tarnovo[-3:]

    
    with open('./artifacts/tarnovo_appartament_price_model.pickle','rb') as f:
        __model_tarnovo = pickle.load(f)
    print('loading saved  Tarnovo artifacts...done')
        
