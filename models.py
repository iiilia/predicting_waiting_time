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
    
    def __init__(self, model, score, stop=False, verbose = False):
        self.verbose = verbose
        self.model = model
        self.models = []
        self.features = []
        self.scores = [sys.float_info.max]
        self.final_model = None
        self.stop = stop
        self.best_iter = None
        self.score = score
        self.name = self.model.__str__().split('(')[0]
        self.main_name = self.__str__().split('(')[0]
    
    def fit(self, df, df_val, features, target):
        # score: less -> better
        
        logging.info('*** '+self.name)
        
        self.all_features = features
        self.flag_stop = True
        
        counter = 0
        
        while self.flag_stop and self.all_features:
            
            counter = counter + 1
            if counter % 10 == 0 and self.verbose:
                sys.stderr.write('-')
                
            res = []
            for feat in self.all_features:
                model = self.model
                model.fit(df[self.features + [feat]],df[target])
                y_pred = model.predict(df_val[self.features + [feat]])
                y_real = df_val[target]
                temp_score = self.score(y_real, y_pred)
                res.append((feat, temp_score, model))
            
            res_sorted = sorted(res, key=lambda x: x[1])  
            
            logging.info('{}/{} : {} - {}'.format(self.main_name, self.name, res_sorted[0][0],res_sorted[0][1]))
            
            if res_sorted[0][1] < self.scores[-1]:
                self.all_features.remove(res_sorted[0][0])
                self.features.append(res_sorted[0][0])
                self.scores.append(res_sorted[0][1])
                self.models.append(res_sorted[0][2])
            else:
                if self.stop:
                    self.flag_stop = False
                else:
                    self.all_features.remove(res_sorted[0][0])
                    self.features.append(res_sorted[0][0])
                    self.scores.append(res_sorted[0][1])
                    self.models.append(res_sorted[0][2])
        
        self.final_model, self.final_score, self.best_iter = sorted(zip(self.scores, 
                                                        self.models,
                                                        range(1,len(self.scores)+1)), key=lambda x: x[0])[0]
        self.final_features = self.features[:self.best_iter]
        
        
    def predict(self, df_test):
        pred = self.final_model.predict(df_test[self.final_features])
        return pred
    
class ExceptFeatures(BaseEstimator, RegressorMixin):
    
    def __init__(self, model, score, stop=False, verbose = False):
        self.verbose = verbose
        self.model = model
        self.models = []
        self.excepted_features = [None]
        self.scores = []
        self.final_model = None
        self.final_score = None
        self.final_iter = None
        self.stop = stop
        self.score = score
        self.name = self.model.__str__().split('(')[0]
        self.main_name = self.__str__().split('(')[0]
        
    def fit(self, df, df_val, features, target):
        # score: less -> better
        logging.info('*** '+self.name )
        
        self.features = features
        self.flag_stop = True
        model = self.model
        model.fit(df[self.features],df[target])
        y_pred = model.predict(df_val[self.features])
        y_real = df_val[target]
        temp_score = self.score(y_real, y_pred)
        self.scores.append(temp_score)
        self.models.append(model)
        counter = 0
        while self.flag_stop and len(set(self.features).difference(set(self.excepted_features)))>0:
            
            counter = counter + 1
            if counter % 10 == 0 and self.verbose:
                sys.stderr.write('-')
            res = []
            
            for feat in set(self.features).difference(set(self.excepted_features)):
                model = self.model
                temp_features = list(set(self.features).difference(set(self.excepted_features)))
                temp_features.remove(feat)
                model.fit(df[temp_features],df[target])
                y_pred = model.predict(df_val[temp_features])
                y_real = df_val[target]
                temp_score = self.score(y_real, y_pred)
                res.append((feat, temp_score, model ))
            
            res_sorted = sorted(res, key=lambda x: x[1])  
            
            logging.info('{}/{} : {} - {}'.format(self.main_name, self.name, res_sorted[0][0],res_sorted[0][1]))
            
            if res_sorted[0][1] < self.scores[-1]:
                self.excepted_features.append(res_sorted[0][0])
                self.scores.append(res_sorted[0][1])
                self.models.append(res_sorted[0][2])
            else:
                if self.stop:
                    self.flag_stop = False
                else:
                    self.excepted_features.append(res_sorted[0][0])
                    self.scores.append(res_sorted[0][1])
                    self.models.append(res_sorted[0][2])
                    
        self.final_model, self.final_score, self.final_iter = sorted(zip(self.scores, 
                                                        self.models,
                                                       range(len(self.scores))), key=lambda x: x[0])[0]
        
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