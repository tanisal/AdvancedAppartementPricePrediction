import json
import pickle
import numpy as np


#-------------Utility Part-----------


#-----Shumen-------

__locations_shumen = None
__data_columns_shumen = None
__build_method_shumen = None
__model_shumen = None

def get_predict_price_shumen(location,m2,rooms,floor,build):
    try:
        loc_index=__data_columns_shumen.index(location.lower())
    except:
        loc_index =-1
    try:
        build_index=__data_columns_shumen.index(build.lower())
    except:
        build_index = -1
    x=np.zeros(len(__data_columns_shumen))
    x[0]=m2
    x[1]=rooms
    x[2]=floor
    if loc_index >= 0:
        x[loc_index] = 1
    if build_index>= 0:
        x[build_index] =1

    return round(__model_shumen.predict([x])[0])


#Function which returns locations list
def get_location_names_shumen():
    return __locations_shumen

#Function which returns building method list
def get_build_method_shumen():
    return __build_method_shumen

#Function which loads server artifacts files, from which we extract locations and build method and model
def load_saved_artifacts_shumen():
    print('loading saved Shumen artifacts...start')
    global __data_columns_shumen
    global __locations_shumen
    global __build_method_shumen
    global __model_shumen

    with open('./artifacts/shumen_columns.json','r',encoding='utf-8') as f:
        __data_columns_shumen = json.load(f)['data_columns']
        __locations_shumen = __data_columns_shumen[3:-3]
        __build_method_shumen = __data_columns_shumen[-3:]

    
    with open('./artifacts/shumen_appartament_price_model.pickle','rb') as f:
        __model_shumen = pickle.load(f)
    print('loading saved  Shumen artifacts...done')
        

#--------------------------------Plovdiv----------------------

__locations_plovdiv = None
__data_columns_plovdiv = None
__build_method_plovdiv = None
__model_plovdiv = None

def get_predict_price_plovdiv(location,m2,rooms,floor,build):
    try:
        loc_index=__data_columns_plovdiv.index(location.lower())
    except:
        loc_index =-1
    try:
        build_index=__data_columns_plovdiv.index(build.lower())
    except:
        build_index = -1
    x=np.zeros(len(__data_columns_plovdiv))
    x[0]=m2
    x[1]=rooms
    x[2]=floor
    if loc_index >= 0:
        x[loc_index] = 1
    if build_index>= 0:
        x[build_index] =1

    return round(__model_plovdiv.predict([x])[0])


#Function which returns locations list
def get_location_names_plovdiv():
    return __locations_plovdiv

#Function which returns building method list
def get_build_method_plovdiv():
    return __build_method_plovdiv

#Function which loads server artifacts files, from which we extract locations and build method and model
def load_saved_artifacts_plovdiv():
    print('loading saved Plovdiv artifacts...start')
    global __data_columns_plovdiv
    global __locations_plovdiv
    global __build_method_plovdiv
    global __model_plovdiv

    with open('./artifacts/plovdiv_columns.json','r',encoding='utf-8') as f:
        __data_columns_plovdiv = json.load(f)['data_columns']
        __locations_plovdiv = __data_columns_plovdiv[3:-3]
        __build_method_plovdiv = __data_columns_plovdiv[-3:]

    
    with open('./artifacts/plovdiv_appartament_price_model.pickle','rb') as f:
        __model_plovdiv = pickle.load(f)
    print('loading saved  Plovdiv artifacts...done')
        
#-----------------------------















if __name__ == '__main__':
    print('Starting Python Flask Server for Real Estate Price Prediction...')
    load_saved_artifacts_shumen()
    print(get_location_names_shumen())
    print(get_build_method_shumen())
    print(get_predict_price_shumen('тракия',58,3,9,'тухла'))
    
    load_saved_artifacts_plovdiv()
    print(get_location_names_plovdiv())
    print(get_build_method_plovdiv())
    print(get_predict_price_plovdiv('център',58,3,9,'тухла'))
    

