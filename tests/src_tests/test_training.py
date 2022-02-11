import pytest, sys
from pathlib import Path
sys.path.insert(1, str((Path(__file__).parent.parent.parent).resolve()))     #path of the project working directory relative to this file
from src.train import *

MODELS=['LGBMRegressor','XGBRegressor','Neural_Network']
@pytest.mark.training
@pytest.mark.parametrize('model', MODELS)
def test_model_creation(model):
    if model == 'LGBMRegressor':
        create_LGBMRegressor_model()
    elif model == 'XGBRegressor':
        create_XGBRegressor_model()
    elif model == 'Neural_Network':
        assert str(type(define_NN_architecture())) == "<class 'keras.engine.sequential.Sequential'>"       #check if the function return a keras model
        create_NN_model()  
        assert open(Path("models") / (model + '.h5'), "rb")  #check if the nn has been saved
    
    #check if the model dictionary is been correctly build  
    assert isinstance(model_wrappers_list[0], dict)         #model_wrappers_list imported from train.py
    assert model_wrappers_list[0]['type'] == model
    assert bool(model_wrappers_list[0]['params'])             #check if the parameter dictionary contains values     
    if model != 'Neural_Network':                             #NN model is not saved in the dictionary due to serialization problem with pickle
        assert str(type(model_wrappers_list[0]['model'])).strip("'>").split('.')[-1] == model        #es. from <'lightgbm.sklearn.LGBMRegressor'> takes LGBMRegressor
    assert type(model_wrappers_list[0]['metrics']) == float 
    
    #check if the model improved or not respect to the last training
    serialized_model_path = Path("models") / (model + '_model.pkl')
    if serialized_model_path.exists():                                  #if is not the first training
        with open(serialized_model_path, "rb") as pickled_model_dict:
           previous_rmse = pickle.load(pickled_model_dict)['metrics']         
        assert model_wrappers_list[0]['metrics'] <= previous_rmse
    model_wrappers_list.clear()      #to have the models always in position 0


@pytest.mark.training
@pytest.mark.parametrize('model', MODELS)
def test_model_serialization(model):
    if model == 'LGBMRegressor':
        create_LGBMRegressor_model()        #create the model dictionary to serialize
    elif model == 'XGBRegressor':
        create_XGBRegressor_model()
    elif model == 'Neural_Network':
        create_NN_model()
    serialize_models_wrappers()
    assert open(Path("models") / (model + '_model.pkl'), "rb")     #check if the serialization created the files
    model_wrappers_list.clear()




