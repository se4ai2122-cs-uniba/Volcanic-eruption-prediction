
from pydantic import BaseModel, Field

# output validation for the 'predict' endpoint
class TimeToErupt(BaseModel):
    eruption_time: str = Field(
        ...,
        description="time to the next eruption starting from the sensors last detection",
        regex='[0-9]+ days, [0-9]+ hours, [0-9]+ minutes, [0-9]+ seconds'
    )

#usefull info for the api endpoint documentation
def get_api_info(type):
        if type == 'description_models':
            return "Available models with some info such as training parameters and Root Mean Squared Error"
        elif type == 'description_predict':
            return "Given a csv of 10 columns with only numeric values, returns day, hour, minutes, seconds until the volcano eruption"
        elif type == 'responses':
          return {
                200: {
                    "description": "Successful response containing day, hour, minutes, seconds until the volcano eruption",
                    "content": {
                        "application/json": {
                            "example": {                   
                                    "message": "OK",
                                    "method": "POST",
                                    "status-code": 200,
                                    "timestamp": "2022-01-19T19:47:47.978372",
                                    "url": "http://localhost:5000/predict/XGBRegressor",
                                    "data": {
                                        "model-type": "XGBRegressor",
                                        "prediction": {
                                        "eruption_time": "22 days, 11 hours, 3 minutes, 6 seconds"
                                        }
                                    }                   
                            }
                        }
                    }
                },
                400: {
                    "description": "Bad request. Model not present or csv not valid ",
                    "content": {
                        "application/json": {
                            "example": {
                                "message": "Model not found, please choose a model available in the models list",
                                "method": "POST",
                                "status-code": 400,
                                "timestamp": "2022-01-19T19:52:14.238496",
                                "url": "http://localhost:5000/predict/XGBRegressork"
                            }
                        }
                    }
                }
            }
        elif type == 'response_model_list':
            return {
                200: {
                    "description": "Successful response containing the models list and their info",
                    "content": {
                        "application/json": {
                            "example": {                   
                                    "message": "OK",
                                    "method": "GET",
                                    "status-code": 200,
                                    "timestamp": "2022-01-19T20:16:10.351636",
                                    "url": "http://localhost:5000/models",
                                    "data": [
                                        {
                                        "type": "LGBMRegressor",
                                        "parameters": {
                                            "num_leaves": 29,
                                            "n_estimators": 189,
                                            "max_depth": 6,
                                            "learning_rate": 0.01,
                                            "random_state": 42
                                        },
                                        "rmse": 8648891.887338564
                                        },
                                        {
                                            "type": "Neural_Network",
                                            "parameters": {
                                                "dropout": 0.6,
                                                "learning_rate": 0.01,
                                                "patience": 20,
                                                "epochs": 1000,
                                                "batch_size": 512,
                                                "activation": "relu"
                                            },
                                            "rmse": 10200079.46856591
                                            },
                                            {
                                            "type": "XGBRegressor",
                                            "parameters": {
                                                "max_depth": 6,
                                                "n_estimators": 189,
                                                "learning_rate": 0.01,
                                                "gamma": 0.788,
                                                "random_state": 42
                                            },
                                            "rmse": 9292197.321645426
                                            }
                                      ]           
                            }
                        }
                    }
                }
            }
        elif type == 'response_root':
            return {
                200: {
                    "description": "Successful response containing the welcome message",
                    "content": {
                        "application/json": {
                            "example": {
                                "message": "OK",
                                "method": "GET",
                                "status-code": 200,
                                "timestamp": "2022-01-19T20:15:47.234203",
                                "url": "http://localhost:5000/",
                                "data": {
                                    "message": "Welcome to Volcanic Eruption Prediction! Please, read the '/docs'!"
                                }
                            }          
                        }
                    }
                }
            }