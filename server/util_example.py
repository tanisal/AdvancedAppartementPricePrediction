

__locations_varna = None
__data_columns_varna = None
__build_method_varna = None
__model_varna = None

def get_predict_price_varna(location,m2,rooms,floor,build):
    try:
        loc_index=__data_columns_varna.index(location.lower())
    except:
        loc_index =-1
    try:
        build_index=__data_columns_varna.index(build.lower())
    except:
        build_index = -1
    x=np.zeros(len(__data_columns_varna))
    x[0]=m2
    x[1]=rooms
    x[2]=floor
    if loc_index >= 0:
        x[loc_index] = 1
    if build_index>= 0:
        x[build_index] =1

    return round(__model_varna.predict([x])[0])


#Function which returns locations list
def get_location_names_varna():
    return __locations_varna

#Function which returns building method list
def get_build_method_varna():
    return __build_method_varna

#Function which loads server artifacts files, from which we extract locations and build method and model
def load_saved_artifacts_varna():
    print('loading saved Varna artifacts...start')
    global __data_columns_varna
    global __locations_varna
    global __build_method_varna
    global __model_varna

    with open('./artifacts/varna_columns.json','r',encoding='utf-8') as f:
        __data_columns_varna = json.load(f)['data_columns']
        __locations_varna = __data_columns_varna[3:-3]
        __build_method_varna = __data_columns_varna[-3:]

    
    with open('./artifacts/varna_appartament_price_model.pickle','rb') as f:
        __model_varna = pickle.load(f)
    print('loading saved  Varna artifacts...done')
