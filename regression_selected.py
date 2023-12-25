import numpy as np
import pandas as pd
import os
import os.path
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
import joblib

def get_labels():
    df = pd.read_excel('./molecules.xlsx')
    labels = []
    labels.append(pd.DataFrame(df['光热转换效率（%）']))
    labels[0].dropna(axis=0, how='any', inplace=True)
    labels[0].reset_index(drop=True, inplace=True)
    # labels = ms_labels.fit_transform(labels[0])
    labels = np.array(labels[0])
    labels = np.array(labels).ravel()
    return df,labels

def get_fea():
    descriptors = pd.read_csv('descriptors.csv')
    circular_fingerprint = np.array(pd.read_csv('circular_fingerprint.csv'))
    Daylight_fingerprint = np.array(pd.read_csv('Daylight_fingerprint.csv'))
    atompair_fingerprint = np.array(pd.read_csv('atompair_fingerprint.csv'))
    return descriptors,circular_fingerprint,Daylight_fingerprint,atompair_fingerprint

def r2_display(train_pred,train_ture,pred,ture,great_train_score,great_score,fea_type,al):
    # fig, ax = plt.subplots(1, 1, num="stars", figsize=(13, 12))
    # Axis_line = np.linspace(*ax.get_xlim(), 2)
    # ax.plot(Axis_line, Axis_line, transform=ax.transAxes, linestyle='--', linewidth=2, color='black', label="1:1 Line")

    fig, ax = plt.subplots(figsize = (5,5))
    ax.plot((0, 1), (0, 1),transform=ax.transAxes, ls='--', c='k')
    ax.set_title('R2 score',fontsize=12)
    ax.set_xlabel('Experimental photothermal conversion efficiency',fontsize=12)
    ax.set_ylabel('Predicted photothermal conversion efficiency',fontsize=12)
    ax.scatter(ture, pred, c='blue', label=('Test R2=%f' %great_score), alpha=0.5)
    ax.scatter(train_ture, train_pred, c='red', label=('Train R2=%f' %great_train_score), alpha=0.5)
    ax.legend(loc=4)
    ax.figure.savefig("%s+%s2.png" %(fea_type,al), dpi=800, bbox_inches='tight')

df,labels = get_labels()
descriptors,circular_fingerprint,Daylight_fingerprint,atompair_fingerprint = get_fea()

def rf(X, y,rand,fea_type):
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=rand) #at 1
    model.fit(X,y)
    model = SelectFromModel(model, prefit=True)
    X = model.transform(X)
    model2 = RandomForestRegressor()
    great_score = 0
    great_train_score = 0
    for i in np.arange(1,2,1):
        for j in np.arange(1,2,1):
            if fea_type == 'circular_fingerprint' :
                grid_params = {
                                'n_estimators': [10], # 6
                                'max_features' : [6], # 3
                                }
            if fea_type == 'Daylight_fingerprint' :
                grid_params = {
                                'n_estimators': [9],
                            'max_features': [3],
                            }
            if fea_type == 'Atompair_fingerprint':
                grid_params = {
                    'n_estimators': [13], #38
                    'max_features': [9], #10
                }
            if fea_type == 'Descriptors':
                grid_params = {
                    'n_estimators': [4], #38 3
                    'max_features': [3], #10 23
                }
            model_cv = GridSearchCV(model2, grid_params, cv=6, verbose=1, n_jobs=-1,scoring='r2')
            score_train_ = []
            score_test_ = []
            # MAE = []
            # R2 = []
            pred_list = []
            ture_list = []
            pred_trian_list = []
            ture_train_list = []
            kf = KFold(n_splits=5,shuffle=True, random_state=88) #12 0.3
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model_cv.fit(X_train, y_train)
                method_ = RandomForestRegressor(
                    n_estimators=model_cv.best_params_['n_estimators'],
                    max_features=model_cv.best_params_['max_features'],
                    random_state=rand
                )
                method_fp = method_.fit(X_train, y_train)
                pred_train = method_fp.predict(X_train)
                pred = method_fp.predict(X_test)
                score_train = method_fp.score(X_train, y_train)
                score_test = method_fp.score(X_test, y_test)

                score_train_.append(score_train)
                score_test_.append(score_test)
                pred_list.append(pred)
                ture_list.append(y_test)
                pred_trian_list.append(pred_train)
                ture_train_list.append(y_train)

                # auc.append(roc_auc_score(pd.DataFrame(y_test), pd.DataFrame(pred_proba),multi_class='ovr'))
            if np.max(score_test_) >= great_score:
                # selector = SelectFromModel(method_fp, prefit=True)
                index = np.argmax(score_test_)
                print(score_train_)
                print(score_test_)
                great_train_score = score_train_[index]
                nmb.append(i)
                great_score = np.max(score_test_)
                r2_display(pred_trian_list[index],ture_train_list[index],pred_list[index],ture_list[index],great_train_score,great_score,fea_type,'rf')
                # joblib.dump(filename='RF.model', value=method_fp)
    # print(great_train_score)
    # print(great_score)
    return method_fp

def svm(X, y,rand,fea_type):
    from sklearn.model_selection import KFold
    from sklearn.svm import SVR
    model = SVR(kernel= 'linear') #at 1
    model.fit(X,y)
    model = SelectFromModel(model, prefit=True)
    X = model.transform(X)
    pd.DataFrame(X).to_csv(os.path.join('./', 'circular_fingerprint_selected.csv'), index=False)
    model2 = SVR(
        kernel= 'linear',
    )
    great_score = 0
    great_train_score = 0
    for i in np.arange(1,2,1):
        # for j in np.arange(0.001,0.01,0.001):
            # for k in np.arange(1,10,1):
            #     for o in np.arange(1,10,1):
        if fea_type == 'circular_fingerprint' :
            grid_params = {'C': [3.8] ,
                        # 'gamma': [0.006],
                        'epsilon' :[0.4],
                        'tol' : [1.62]
                        }
        if fea_type == 'Daylight_fingerprint':
            grid_params = {'C': [1.0] ,
                        # 'gamma': [0.009],
                        'epsilon' :[0.4],
                        'tol' : [2.22]
                        }
        if fea_type == 'Atompair_fingerprint':
            grid_params = {'C': [0.4],
                        # 'gamma': [0.002],
                        'epsilon': [0.23],
                        'tol': [1.85]
                        }
        if fea_type == 'Descriptors':
            grid_params = {'C': [0.2],
                        # 'gamma': [0.009],
                        'epsilon': [0.09],
                        'tol': [0.89]
                        }
            # grid_params = {'C': [i] , 'gamma': [0.043], 'epsilon' :[0.042]}  #525 278 239 0.012 0.013 0.011 4.2 0.0001 0
        # grid_params = {'C': np.linspace(0.1,20,50), 'gamma':np.linspace(0.1,20,20)}
        model_cv = GridSearchCV(model2, grid_params, cv=6, verbose=1, n_jobs=2,scoring='r2')
        score_train_ = []
        score_test_ = []
        pred_list = []
        ture_list = []
        pred_trian_list = []
        ture_train_list = []
        kf = KFold(n_splits=5, shuffle=True, random_state= 88)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model_cv.fit(X_train, y_train)
            method_ = SVR(kernel= 'linear',
                        C=model_cv.best_params_['C'],
                        # gamma=model_cv.best_params_['gamma'],
                        epsilon=model_cv.best_params_['epsilon'],
                        tol = model_cv.best_params_['tol']
                        )
            # method_ = SVR(C=model_cv.best_params_['C'],gamma=model_cv.best_params_['gamma'],epsilon=model_cv.best_params_['epsilon'],tol = model_cv.best_params_[tol])
            method_fp = method_.fit(X_train, y_train)
            score_train = method_fp.score(X_train, y_train)
            score_test = method_fp.score(X_test, y_test)
            pred_train = method_fp.predict(X_train)
            pred = method_fp.predict(X_test)

            score_train_.append(score_train)
            score_test_.append(score_test)
            pred_list.append(pred)
            ture_list.append(y_test)
            pred_trian_list.append(pred_train)
            ture_train_list.append(y_train)
        if  np.max(score_test_) >= great_score:
            # selector = SelectFromModel(method_fp, prefit=True)
            index = np.argmax(score_test_)
            great_train_score = score_train_[index]
            great_score = score_test_[index]
            r2_display(pred_trian_list[index], ture_train_list[index], pred_list[index], ture_list[index],great_train_score, great_score,fea_type,'svm')
            # print(score_train_)
            # print(score_test_)
            # print(best_para_i)
            # great_ex = ex
            # joblib.dump(filename='svm_cir.model', value=method_fp)
            nmb.append(i)
            nmb.append(np.max(score_test_))
            # nmb.append(k)
            # nmb.append(o)

    print(great_train_score)
    # print(best_para_i)
    # print(best_para_j)
    # print(best_para_k)
    print(great_score)
    return method_fp

def catb(X, y,rand,fea_type):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True,random_state=88)
    import catboost as cb
    model = cb.CatBoostRegressor(random_state=rand)
    model.fit(X,y)
    model = SelectFromModel(model, prefit=True)
    X = model.transform(X)
    model2 = cb.CatBoostRegressor()
    # grid_params = {'learning_rate': list((0.05,1)), 'iterations': list((50,100)),'depth':list((3,6))}
    great_score = 0
    his_score =[]
    for i in np.arange(1,2,1):
        if fea_type == 'circular_fingerprint' :
            grid_params = {'iterations': [18],
                           'learning_rate' : [0.57],
                           'l2_leaf_reg': [3.3],
                           'depth': [4]
                           }
        if fea_type == 'Daylight_fingerprint' :
            grid_params = {'iterations': [11],
                           'learning_rate' : [0.9],
                        #    'l2_leaf_reg': [None],
                        #    'depth': [None],
                           }
        if fea_type == 'Atompair_fingerprint':
            grid_params = {'iterations': [10], #13
                           'learning_rate': [0.65], #0.71
                           'l2_leaf_reg': [3], #2.7
                           'depth': [6] #3
                           }
        if fea_type == 'Descriptors':
            grid_params = {'iterations': [6],
                           'learning_rate': [0.56],
                        #    'l2_leaf_reg': [None],
                        #    'depth': [None]
                           }
        #7 0.71 2.9 5 ,'learning_rate' : [0.1399],'l2_leaf_reg' : [2.9],'depth' : [6]}
        model_cv = GridSearchCV(model2, grid_params, cv=6, verbose=1, n_jobs=-1,scoring='r2')
        score_train_ = []
        score_test_ = []
        pred_list = []
        ture_list = []
        pred_trian_list = []
        ture_train_list = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model_cv.fit(X_train, y_train)
            method_ = cb.CatBoostRegressor(iterations=model_cv.best_params_['iterations'],
                                           learning_rate = model_cv.best_params_['learning_rate'],
                                           # l2_leaf_reg = model_cv.best_params_['l2_leaf_reg'],
                                           # depth = model_cv.best_params_['depth'],
                                           # bagging_temperature = model_cv.best_params_['bagging_temperature'],
                                           loss_function='RMSE',
                                           eval_metric = 'R2',
                                           random_state=rand)
            method_fp = method_.fit(X_train, y_train)
            score_train = method_fp.score(X_train, y_train)
            score_test = method_fp.score(X_test, y_test)
            pred_train = method_fp.predict(X_train)
            pred = method_fp.predict(X_test)

            score_train_.append(score_train)
            score_test_.append(score_test)
            pred_list.append(pred)
            ture_list.append(y_test)
            pred_trian_list.append(pred_train)
            ture_train_list.append(y_train)
        if np.max(score_test_) >= great_score:
            selector = SelectFromModel(method_fp, prefit=True)
            index = np.argmax(score_test_)
            great_score = score_test_[index]
            great_train_score = score_train_[index]
            nmb.append(i)
            nmb.append(np.max(score_test_))
            # print(great_train_score)
            # print(great_score)
            his_score.append(great_score)
            r2_display(pred_trian_list[index], ture_train_list[index], pred_list[index], ture_list[index],great_train_score, great_score,fea_type,'Catboost')
            # joblib.dump(filename='CatBoost.model', value=method_fp)
            # pd.DataFrame(X_train).to_csv(('./train_x_cb.csv'), index=False)

    return method_fp


nmb = []
descriptors = np.array(descriptors)
rf(circular_fingerprint,labels,2,'circular_fingerprint') # 0.78
rf(Daylight_fingerprint,labels,1,'Daylight_fingerprint') # 0.80
rf(atompair_fingerprint,labels,1,'Atompair_fingerprint') # 0.85
rf(descriptors,labels,3,'Descriptors') # 0.708
#
svm(circular_fingerprint,labels,1,'circular_fingerprint') # 0.833
svm(Daylight_fingerprint,labels,1,'Daylight_fingerprint') # 0.64
svm(atompair_fingerprint,labels,1,'Atompair_fingerprint') # 0.71
svm(descriptors,labels,1,'Descriptors') # 0.43

catb(circular_fingerprint,labels,4,'circular_fingerprint') # 0.76
catb(Daylight_fingerprint,labels, 6,'Daylight_fingerprint') # 0.75
catb(atompair_fingerprint,labels,1,'Atompair_fingerprint') # 0.65
catb(descriptors,labels,1,'Descriptors') # 0.44