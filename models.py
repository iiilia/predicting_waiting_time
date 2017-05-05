import pandas as pd
import numpy as np
import pickle as pkl
import os, gc, math, sys, re
import copy as cp
import numpy as np
import logging
import pandas as pd
import time
import threading

from tqdm import *
from datetime import datetime as dt
from datetime import datetime, timedelta
from multiprocessing import Pool
from multiprocessing import Process
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
    
    def __init__(self, model, scaler, scores_f, score, stop=False, verbose = False):
        self.verbose = verbose
        self.model = model
        self.models = []
        self.scaler = scaler
        self.features = []
        self.scores = [sys.float_info.max]
        self.final_model = None
        self.stop = stop
        self.best_iter = None
        self.score = score
        self.scores_f = scores_f
        self.name = self.model.__str__().split('(')[0]
        self.main_name = self.__str__().split('(')[0]
    
    @staticmethod
    def f_train(model_, df, df_val, features, target, feat, scores_f, ind, res):
        model = model_
        model.fit(df[features + [feat]].values,df[target].values)
        y_pred = model.predict(df_val[features + [feat]].values)
        y_real = df_val[target]
        sc = {}
        for score_ in scores_f:
            sc[score_] = scores_f[score_](y_real, y_pred)    
        res[ind] = (feat, sc, model)
        
    def fit(self, df, df_val, features, target):
        # score: less -> better
        
        logging.info('*** '+self.name)
        df_target = df[target].copy()
        df_val_target = df_val[target].copy()
        
        df = pd.DataFrame(self.scaler.fit_transform(df[features].astype(float)), columns=features)
        df_val = pd.DataFrame(self.scaler.transform(df_val[features].astype(float)), columns=features)
        df[target] = df_target
        df_val[target] = df_val_target
        
        self.all_features = features
        self.flag_stop = True
        
        counter = 0
        
        while self.flag_stop and len(self.all_features)>0:
            
            counter = counter + 1
            if counter % 10 == 0 and self.verbose:
                sys.stderr.write('-')
                
            sys.stderr.write('-')    
            
            temp_all_features = self.all_features
            
            ###
            res = [None]*len(temp_all_features)

            threads = [None]*len(temp_all_features)

            for ind, feat in enumerate(temp_all_features):
                m = self.model
                threads[ind] = threading.Thread(target=AddFeatures.f_train,
                #threads[ind] = Process(target=AddFeatures.f_train,
                                                name="proc_"+str(ind),
                                                args=[cp.deepcopy(m),
                                                      df,
                                                      df_val,
                                                      cp.deepcopy(self.features), 
                                                      cp.deepcopy(target), 
                                                      cp.deepcopy(feat),
                                                      self.scores_f,
                                                      ind, 
                                                      res])
                threads[ind].start()

            for ind in range(len(threads)):
                threads[ind].join()
                
            ###   
            res_sorted = sorted(res, key=lambda x: x[1][self.score])  
            
            logging.info('{}/{} : {} - {}'.format(self.main_name, self.name, res_sorted[0][0],res_sorted[0][1]))
            
            if res_sorted[0][1] < self.scores[-1]:
                try:
                    temp_all_features.remove(res_sorted[0][0])
                    self.all_features = temp_all_features
                except ValueError as e:
                    logging.warning(u'WARNING: We have not "{}" feature'.format(str(res_sorted[0][0])))
                self.features.append(res_sorted[0][0])
                self.scores.append(res_sorted[0][1])
                self.models.append(res_sorted[0][2])
            else:
                if self.stop:
                    self.flag_stop = False
                else:
                    try:
                        temp_all_features.remove(res_sorted[0][0])
                        self.all_features = temp_all_features
                    except ValueError as e:
                        logging.warning(u'WARNING: We have not "{}" feature'.format(str(res_sorted[0][0])))
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
    
    def __init__(self, model, scaler, scores_f, score, stop=False, verbose = False):
        self.verbose = verbose
        self.model = model
        self.models = []
        self.scaler = scaler
        # excluded features
        self.excepted_features = [None]
        self.scores = []
        self.final_model = None
        self.final_score = None
        self.final_iter = None
        self.stop = stop
        self.score = score
        self.scores_f = scores_f
        self.name = self.model.__str__().split('(')[0]
        self.main_name = self.__str__().split('(')[0]
        
    @staticmethod
    def f_train(model_, df, df_val, temp_features, target, scores_f, ind, res):
        model = model_
        temp_features.remove(feat)
        model.fit(df[temp_features].values,df[target].values)
        y_pred = model.predict(df_val[temp_features].values)
        y_real = df_val[target]
        sc = {}
        for score_ in scores_f:
            sc[score_] = scores_f[score_](y_real, y_pred)    
        res[ind] = (feat, sc, model)
        
    def fit(self, df, df_val, features, target):
        # score: less -> better
        logging.info('*** '+self.name )
        df_target = df[target].copy()
        df_val_target = df_val[target].copy()
        df = pd.DataFrame(self.scaler.fit_transform(df[features].astype(float)), columns=features)
        df_val = pd.DataFrame(self.scaler.transform(df_val[features].astype(float)), columns=features)
        df[target] = df_target
        df_val[target] = df_val_target
        
        # all features
        self.features = features
        self.flag_stop = True
        model = self.model
        model.fit(df[self.features],df[target])
        y_pred = model.predict(df_val[self.features])
        y_real = df_val[target]
        sc = {}
        for score_ in self.scores_f:
            sc[score_] = self.scores_f[score_](y_real, y_pred)    
        self.scores.append(sc)
        self.models.append(model)
        
        counter = 0
        while self.flag_stop and len(set(self.features).difference(set(self.excepted_features)))>0:
            
            counter = counter + 1
            if counter % 10 == 0 and self.verbose:
                sys.stderr.write('-')
            
            sys.stderr.write('-')
            temp_features = list(set(self.features).difference(set(self.excepted_features)))
            ###
            res = [None]*len(temp_features)
            threads = [None]*len(temp_features)

            for ind, feat in enumerate(temp_features):
                m = self.model
                threads[ind] = threading.Thread(target=AddFeatures.f_train,
                #threads[ind] = Process(target=AddFeatures.f_train,
                                                name="proc_"+str(ind),
                                                args=[cp.deepcopy(m),
                                                      df,
                                                      df_val,
                                                      cp.deepcopy(temp_features), 
                                                      cp.deepcopy(target), 
                                                      cp.deepcopy(feat),
                                                      self.scores_f,
                                                      ind, 
                                                      res])
                threads[ind].start()

            for ind in range(len(threads)):
                threads[ind].join()
            ###   
            res_sorted = sorted(res, key=lambda x: x[1][self.score]) 
            
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
                logging.warning(u'WARNING: We have not {} feature'.format(str(i)))
        self.final_features = temp_features
        
    def predict(self, df_test):
        pred = self.final_model.predict(df_test[self.final_features])
        return pred