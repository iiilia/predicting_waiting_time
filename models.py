import pandas as pd
import numpy as np
import pickle as pkl
import os, gc, math, sys, re
import os,sys
import numpy as np
import logging
import pandas as pd
import time

from tqdm import *
from datetime import datetime as dt
from datetime import datetime, timedelta
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.cluster import KMeans
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

class AddFeatures(BaseEstimator, RegressorMixin):
    
    def __init__(self, model, stop=False):
        self.model = model
        self.models = []
        self.features = []
        self.score = [sys.float_info.max]
        self.final_model = None
        self.stop = stop
        self.best_iter = None
        logging.info(self.model.__class__)
    
    def fit(self, df, df_val, features, target, score):
        # score: less -> better
        
        self.all_features = features
        self.flag_stop = True
        while self.flag_stop and self.all_features:
            res = []
            for feat in self.all_features:
                model = self.model
                model.fit(df[self.features + [feat]],df[target])
                y_pred = model.predict(df_val[self.features + [feat]])
                y_real = df_val[target]
                temp_score = score(y_real, y_pred)
                res.append((feat, temp_score, model))
            
            res_sorted = sorted(res, key=lambda x: x[1])  
            
            logging.info('{} - {}'.format(res_sorted[0][0],res_sorted[0][1]))
            
            if res_sorted[0][1] < self.score[-1]:
                self.all_features.remove(res_sorted[0][0])
                self.features.append(res_sorted[0][0])
                self.score.append(res_sorted[0][1])
                self.models.append(res_sorted[0][2])
            else:
                if self.stop:
                    self.flag_stop = False
                else:
                    self.all_features.remove(res_sorted[0][0])
                    self.features.append(res_sorted[0][0])
                    self.score.append(res_sorted[0][1])
                    self.models.append(res_sorted[0][2])
        
        self.final_model, self.final_score, self.best_iter = sorted(zip(self.score, 
                                                        self.models,
                                                        range(1,len(self.score)+1)), key=lambda x: x[0])[0]
        self.final_features = self.features[:self.best_iter]
        
        
    def predict(self, df_test):
        pred = self.final_model.predict(df_test[self.final_features])
        return pred
    
class ExceptFeatures(BaseEstimator, RegressorMixin):
    
    def __init__(self, model, stop=False):
        self.model = model
        self.models = []
        self.excepted_features = [None]
        self.score = []
        self.final_model = None
        self.final_score = None
        self.final_iter = None
        self.stop = stop
        logging.info(self.model.__class__)
    
    def fit(self, df, df_val, features, target, score):
        # score: less -> better
        
        self.features = features
        self.flag_stop = True
        model = self.model
        model.fit(df[self.features],df[target])
        y_pred = model.predict(df_val[self.features])
        y_real = df_val[target]
        temp_score = score(y_real, y_pred)
        self.score.append(temp_score)
        self.models.append(model)
        
        while self.flag_stop and len(set(self.features).difference(set(self.excepted_features)))>0:
            res = []
            for feat in set(self.features).difference(set(self.excepted_features)):
                model = self.model
                temp_features = list(set(self.features).difference(set(self.excepted_features)))
                temp_features.remove(feat)
                model.fit(df[temp_features],df[target])
                y_pred = model.predict(df_val[temp_features])
                y_real = df_val[target]
                temp_score = score(y_real, y_pred)
                res.append((feat, temp_score, model ))
            
            # sys.stderr.write('|||'.join(self.excepted_features))
            res_sorted = sorted(res, key=lambda x: x[1])  
            
            logging.info('{} - {}'.format(res_sorted[0][0],res_sorted[0][1]))
            
            if res_sorted[0][1] < self.score[-1]:
                self.excepted_features.append(res_sorted[0][0])
                self.score.append(res_sorted[0][1])
                self.models.append(res_sorted[0][2])
            else:
                if self.stop:
                    self.flag_stop = False
                else:
                    self.excepted_features.append(res_sorted[0][0])
                    self.score.append(res_sorted[0][1])
                    self.models.append(res_sorted[0][2])
                    
        self.final_model, self.final_score, self.final_iter = sorted(zip(self.score, 
                                                        self.models,
                                                       range(len(self.score))), key=lambda x: x[0])[0]
        
        temp_features = self.features
        for i in self.excepted_features[:self.best_iter]:
            try:
                temp_features.remove(i)
            except ValueError as e:
                logging.warning(u'We have not {} feature'.format(str(i)))
        self.final_features = temp_features
        
    def predict(self, df_test):
        pred = self.final_model.predict(df_test[self.final_features])
        return pred