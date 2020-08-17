
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 

# basic modules
from datetime import datetime
import numpy as np
import pandas as pd
import math, sys, argparse
import time

# preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn. neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, BernoulliRBM

# ensemble
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

# pipeline
# from sklearn.pipeline import Pipeline, make_pipeline


# model selection
Ffrom sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold, RepeatedStratifiedKFold, RepeatedKFold

# calibration
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# metrics scores
from sklearn.metrics import roc_auc_score, average_precision_score, make_scorer, f1_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

# feature selection
from sklearn.feature_selection import RFECV

# external modules
# sys.path.append('/home/FORDHAM/tdoan5/ext_modules/mlxtend-0.13.0-py2.py3-none-any.whl')
# Azure server has xgboost module already, but do not have latest sklearn version
# Erdos server do not have mlxtend, xgboost
#sys.path.append('/u/erdos/csga/doan/ext_modules/xgboost-0.72.1-py2.py3-none-manylinux1_x86_64.whl')
from xgboost import XGBClassifier
import catboost as cb
import lightgbm as lgb
# sys.path.append('/u/erdos/csga/doan/ext_modules/automl-2.9.9-py2.py3-none-any.whl')
# automl-2.9.9-py2.py3-none-any.whl

#sys.path.append('/u/erdos/csga/doan/ext_modules/mlxtend-0.13.0-py2.py3-none-any.whl')

#from mlxtend.classifier import StackingCVClassifier

#sys.path.append('/u/erdos/csga/doan/ext_modules/imbalanced_learn-0.3.3-py3-none-any.whl')
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedBaggingClassifier
np.set_printoptions(precision=3)

print(__doc__)


def parse_arguments():
    
    # Instantiate the parser
    parser = argparse.ArgumentParser(description=
        """
        algr (required) is in range 1-20:\n

        1: ('knn', KNeighborsClassifier()),
        2: ('lr', LogisticRegression()),
        3: ('dt', DecisionTreeClassifier()),
        4: ('xtr', ExtraTreesClassifier()),
        5: ('rf', RandomForestClassifier()),
        6: ('gbt', GradientBoostingClassifier()),
        7: ('mlp', MLPClassifier()),
        8: ('bnb', BernoulliNB()),
        9: ('gnb', GaussianNB()),
        10: ('polysvc', SVC()),
        11: ('sigmsvc', SVC()),
        12: ('rbfsvc', SVC()),
        13: ('lsvc', SVC()),
        14: ('lbsvc', LinearSVC()),
        15: ('bsvc', BalancedBaggingClassifier()),
        16: ('absvc', AdaBoostClassifier()),
        17: ('ccsvc', CalibratedClassifierCV()),
        18: ('bbnb', BalancedBaggingClassifier()),
        19: ('blsvc', BalancedBaggingClassifier()),
        20: ('blr', BalancedBaggingClassifier()),
        21: ('bdt', BalancedBaggingClassifier()),
        22: ('blsvc', BalancedBaggingClassifier()),
        """
    )

    # Required positional argument
    parser.add_argument('algr', type=int,
                        help='A required integer positional argument')

    # Optional positional argument
    parser.add_argument('resample', type=int, nargs='?',
                        help='An optional integer positional argument: 0 = no sampling (default), 1 = SMOTE imblearn, 2 = pandas.sample')

    # Optional argument
    parser.add_argument('--kfolds', type=int,
                        help='An optional integer argument: None (default kfolds=10), else user-defined 5 or 10-folds')

    # Optional argument
    parser.add_argument('--numtrial', type=int,
                        help='An optional integer argument: None (default kfolds=2)')

    # Optional argument
    parser.add_argument('--threshold', type=float,
                        help='An optional float argument: None (default threshold=0.5), else user-defined 0 < threshold < 1')

    # Seed
    parser.add_argument('--seed', type=int,
                        help='An optional integer argument to provide seed for shuffling data')

    return parser

def check_parser():
    # python filename.py 1 2 --kfolds 3 --switch
    parser = parse_arguments()
    args = parser.parse_args()
    """
    print("Argument values:")
    print(args.algr)
    print(args.resample)
    print(args.kfolds)
    print(args.switch)
    """
    if args.algr not in list(np.arange(1,35)):
        parser.error(
            """
            'algr' must in range 1-10:

            1: ('knn', KNeighborsClassifier()),
            2: ('lr', LogisticRegression()),
            3: ('dt', DecisionTreeClassifier()),
            4: ('xtr', ExtraTreesClassifier()),
            5: ('rf', RandomForestClassifier()),
            6: ('gbt', GradientBoostingClassifier()),
            7: ('mlp', MLPClassifier()),
            8: ('bnb', BernoulliNB()),
            9: ('gnb', GaussianNB()),
            10: ('polysvc', SVC()),
            11: ('sigmsvc', SVC()),
            12: ('rbfsvc', SVC()),
            13: ('lsvc', SVC()),
            14: ('lbsvc', LinearSVC()),
            15: ('bsvc', BalancedBaggingClassifier()),
            16: ('absvc', AdaBoostClassifier()),
            17: ('ccsvc', CalibratedClassifierCV()),
            18: ('bbnb', BalancedBaggingClassifier()),
            19: ('blsvc', BalancedBaggingClassifier()),
            20: ('bsvc', BaggingClassifier()),
            21: ('bdt', BalancedBaggingClassifier()),
            22: ('blsvc', BalancedBaggingClassifier()),
            """
        )

    if args.resample not in [None, 0 , 1, 2]:
        parser.error(
            """
            0 : no sampling
            1 : SMOTE imblearn
            2 : pandas.sample
            """
        )
    if args.threshold == None: pass
    elif args.threshold<0 or args.threshold>1:
        parser.error("threhold must be between 0 and 1")

    """
    if args.kfolds not in [5, 10, None]:
        parser.error("kfolds can be either 5 or 10 cross-validations")
    """

    return parser

def compute_measure(true_label, predicted_label):
    tn,fp,fn,tp = confusion_matrix(true_label, predicted_label).ravel()
    tn = np.array(tn)
    fp = np.array(fp)
    fn = np.array(fn)
    tp = np.array(tp)
    
    with np.errstate(divide='ignore'):
        sen=(1.0*tp)/(tp+fn)
        
    with np.errstate(divide='ignore'):
        spc=(1.0*tn)/(tn+fp)
    
    with np.errstate(divide='ignore'):
        ppr=(1.0*tp)/(tp+fp)
    
    with np.errstate(divide='ignore'):
        npr=(1.0*tn)/(tn+fn)    
    

    acc=(tp+tn)*1.0/(tp+fp+tn+fn)

    #diag=math.log(1+acc,2)+math.log(1+(sen+spc)/2,2)
  
    sen=np.around(sen,2)
    spc=np.around(spc,2)
    ppr=np.around(ppr,2)
    npr=np.around(npr,2)
    acc=np.around(acc,2)
    # diag=round(diag,2)
    
    ans=[]
    ans.append(acc)
    ans.append(sen)
    ans.append(spc)
    ans.append(ppr)
    ans.append(npr)
    # ans.append(diag)    
    return ans

def plot_roc_curve(fpr, tpr, roc_auc, pos=1):
    plt.figure()
    lw = 2
    plt.plot(fpr[pos], tpr[pos], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[pos])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR (1 - spec)')
    plt.ylabel('TPR (recall/sens)')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
   
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])    


def cutoff_predict(clf, X, threshold):
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    return (prob_pos>=threshold).astype(int)

def clf_report(clf, X, y):
    y_pred = clf.predict(X)
    print(compute_measure(y, y_pred))
    print(classification_report(y, y_pred))

def cutoff_predict_proba(clf, X):
    # use this only for X_test to have same index for each sample.
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    return prob_pos

def custom_f1(threshold):
    def f1_cutoff(clf, X, y):
        y_pred = cutoff_predict(clf, X, threshold)
        return f1_score(y, y_pred)
    return f1_cutoff

def resampling(X, y):

    # resample minority class (run on Erdos without imblearn SMOTE)
    Xc = pd.concat([X, y], axis=1)
    count_class_0, count_class_1 = Xc['labels'].value_counts()

    # Divide by class
    df_class_0 = Xc[Xc['labels'] == 0]
    df_class_1 = Xc[Xc['labels'] == 1]
    
    df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    df_res = pd.concat([df_class_0, df_class_1_over], axis=0)
    Xres = df_res.iloc[:,:df_res.shape[1]]
    yres = df_res.iloc[:,df_res.shape[1]]
    return Xres, yres

def smote_resample(X,y):
    sm = SMOTE()
    X_res, y_res = sm.fit_sample(X, y)
    return X_res, y_res

def switch_algorithm(algr):
    switcher = {
        1: ('knn', KNeighborsClassifier()),
        2: ('lr', LogisticRegression(solver='liblinear')),
        3: ('dt', DecisionTreeClassifier()),
        4: ('xtr', ExtraTreesClassifier()),
        5: ('rf', RandomForestClassifier()),
        6: ('gbt', GradientBoostingClassifier()),
        7: ('mlp', MLPClassifier()),
        8: ('bnb', BernoulliNB()),
        9: ('gnb', GaussianNB()),
        10: ('polysvc', SVC()),
        11: ('sigmsvc', SVC()),
        12: ('rbfsvc', SVC()),
        13: ('lsvc', SVC()),
        14: ('lbsvc', LinearSVC()),
        15: ('bsvc', BalancedBaggingClassifier(SVC(kernel='linear', probability=True), sampling_strategy='not majority')),
        16: ('absvc', BalancedBaggingClassifier(SVC(kernel='linear', probability=True), sampling_strategy='all')),
        17: ('ccsvc', CalibratedClassifierCV()),
        18: ('bbnb', BalancedBaggingClassifier()),
        19: ('blsvc', BalancedBaggingClassifier()),
        20: ('bsvc',BalancedBaggingClassifier()),
        21: ('bsvcsig', BalancedBaggingClassifier()),
        22: ('xgbt', XGBClassifier(n_thread=-1)),
        23: ('bxgbt', BalancedBaggingClassifier(XGBClassifier(n_thread=-1))),
        24: ('bgbt', BalancedBaggingClassifier(GradientBoostingClassifier())),
        25: ('adb', AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, class_weight='balanced'))),
        26: ('lgbm', lgb.LGBMClassifier(silent=True, class_weight='balanced')),
        27: ('catb', cb.CatBoostClassifier(silent=True))
    }
    # print(switcher.get(algr, "Invalid algorithm"))
    return switcher.get(algr, "Invalid algorithm")

def switch_paramsGrid(p_grid):
    switcher = {
        1: {
            'knn__n_neighbors': range(3, 15, 2),
            'knn__weights': ["uniform", "distance"],
            'knn__p': [1, 2]
        },
        2: {
            #'lr__C' : np.arange(0.01, 0.1, 0.01),
            'lr__C' : [0.07, 0.08, 0.09],
            'lr__penalty': ['l1'],
            'lr__class_weight': [{0:1,1:4}]
        },
        3: {
            'dt__criterion': ['entropy', 'gini'],
            'dt__max_features': [None, 'auto'],
            'dt__max_depth' : [2],
            'dt__class_weight' : [{0:1, 1:3}]
        },
        4: {
            'xtr__n_estimators': [50],
            'xtr__criterion': ["gini", "entropy"],
            'xtr__max_depth' : [3],
            'xtr__bootstrap': [True],
            'xtr__class_weight' : [{0:1,1:4}]
        },
        5: {
            'rf__n_estimators': [25],
            'rf__criterion': ["gini", "entropy"],
            'rf__max_features': np.arange(0.05, 1.01, 0.1),
            'rf__max_depth' : [2],
            'rf__bootstrap': [True],
            'rf__class_weight' : [{0:1, 1:4}]
        },
        6: {
            'gbt__n_estimators': [50],
            'gbt__criterion': ["friedman_mse"],
            'gbt__max_depth' : [1],
            #'gbt__min_samples_split': range(2, 5),
            #'gbt__min_samples_leaf': range(1, 5),
        },
        7: {
            'mlp__hidden_layer_sizes': [(1,)],
            'mlp__activation' : ['logistic'],
            'mlp__alpha' : [1e-3, 1e-2],
            'mlp__max_iter' : [1000],
            'mlp__learning_rate' : ['adaptive']
        },
        8: {
            'bnb__alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
            'bnb__fit_prior': [True, False]
        },
        9: {
            'gnb__priors': [None, [0.2, 0.8], [0.3, 0.7]]
        },
        10: {
            'polysvc__kernel' : ['poly'],
            'polysvc__degree' : [2],
            'polysvc__gamma': [1e-4],
            'polysvc__coef0' : [-5, 0, 5],
            'polysvc__tol': [1e-4],
            'polysvc__C': [1.],
            'polysvc__class_weight' : ['balanced'],
            'polysvc__probability' : [True]
        },
        11: {
            'sigmsvc__kernel' : ['sigmoid'],
            'sigmssvc__gamma': [1e-4, 1e-3, 1e-2],
            'sigmsvc__tol': [1e-4, 1e-3],
            'sigmsvc__coef0' : [-5, 0, 5],
            'sigmsvc__C': [1, 10],
            'sigmsvc__class_weight' : ['balanced', {0:1,1:5}, {0:1,1:7}],
            'sigmsvc__probability' : [True]
        },
        12: {
            'rbfsvc__kernel' : ['rbf'],
            'rbfsvc__gamma': [1e-3, 1e-2, 1e-1, 1],
            'rbfsvc__tol': [1e-4],
            'rbfsvc__C': [1e-2, 1e-1, 1, 10],
            'rbfsvc__class_weight' : ['balanced'],
            'rbfsvc__probability' : [True]
        },
        13: {
            'lsvc__kernel' : ['linear'],
            'lsvc__loss' : ['hinge', 'squared_hinge'],
            'lsvc__dual' : [True, False],
            'lsvc__tol': [1e-4, 1e-3],
            'lsvc__C': [1e-2, 1e-1, 1., 1, 10],
            'lsvc__class_weight' : ['balanced', {0:1,1:5}, {0:1,1:7}, {0:1,1:3}],
            'lsvc__probability' : [True]
        },
        14: {
            'lbsvc__loss' : ['hinge', 'squared_hinge'],
            'lbsvc__tol': [1e-4, 1e-3],
            'lbsvc__C': [1e-2, 1e-1, 1., 1, 10],
            'lbsvc__class_weight' : ['balanced', {0:1,1:5}, {0:1,1:7}, {0:1,1:3}]
        },
        15: {
            #'bsvc__base_estimator' : [SVC(C=1, kernel='linear', probability=True, class_weight={0:1,1:5})],
            'bsvc__base_estimator__C' : [1],
            'bsvc__n_estimators': [50],
        },
        16: {
            'absvc__base_estimator__C' : [1],
            'absvc__n_estimators': [50],

        },
        17: {
            'ccsvc__base_estimator' : [SVC(C=10, kernel='sigmoid')],
            'ccsvc__method': ['sigmoid', 'isotonic'],
            'ccsvc__cv': [2, 3]
        },
        18: {
            'bbnb__base_estimator': [BernoulliNB()],
            'bbnb__n_estimators': [10, 20, 50, 100]
        },
        19: {
            'blsvc__base_estimator' : [LinearSVC(C=1, class_weight=None, max_iter=10000)],
            'blsvc__n_estimators': [50],
        },
        20: {
            'bsvc__base_estimator' : [SVC(C=1, kernel='linear', probability=True, class_weight={0:1,1:5})],
            'bsvc__n_estimators': [50, 100],
        },
        21: {
            'bsvcsig__base_estimator' : [SVC(C=10, kernel='sigmoid', probability=True, class_weight={0:1,1:5})],
            'bsvcsig__n_estimators': [20, 50, 100],
        },
        22: {
            'xgbt__max_depth' : [1],
            'xgbt__scale_pos_weight': [1], 
        },
        23: {
            'bxgbt__base_estimator__max_depth' : [1],
        },
        24: {
            'bgbt__base_estimator__max_depth' : [1],
        },
        25: {
            'adb__base_estimator__max_depth' : [1],
        },
        26: {
            'lgbm__max_depth' : [1],
            'lgbm__class_weight' : [None],
            'lgbm__learning_rate' : [0.01, .1],
            'lgbm__n_estimators' : [200]
        },
        27: {
            'catb__depth' : [1],
            'catb__class_weights' : [[1, 1]],
            'catb__learning_rate' : [0.1],
            'catb__l2_leaf_reg' : [1],
            'catb__iterations' : [100]

        }
    }
    return switcher.get(p_grid, "Invalid params grid")

def avg_pr_auc_score(y_true, y_score):        
    return average_precision_score(y_true, y_score, average='weighted')

def pr_auc_score(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

def roc_auc_func(y_true, y_score):
    return roc_auc_score(y_true, y_score, average='weighted')

def scoring_refit_gridseachCV(refit=0):
    scoring = {
        'roc_auc_w': make_scorer(roc_auc_func),
        'avg_prec_w' : make_scorer(avg_pr_auc_score),
        'pr_auc': make_scorer(pr_auc_score)
    }
    if refit==0: gs_refit = 'roc_auc_w'
    if refit==1: gs_refit = 'avg_prec_w'
    if refit==2: gs_refit = 'pr_auc'
    return scoring, gs_refit

def print_res(fpr, tpr, thresholds, class_1_min_recall=.60):
    sens  = tpr[tpr>=class_1_min_recall]
    specs = 1 - fpr[tpr>=class_1_min_recall]
    thres = thresholds[tpr>=class_1_min_recall]

    print("Threshold: ")
    print(thres[0])

    print("Recall class 0:")
    print(specs[0])


    print("Recall class 1:")
    print(sens[0])

def load_data(seed, shuffl=False):

    X = pd.read_csv('D1_X.csv', dtype=np.float64)
    y = pd.read_csv('Ys.csv', dtype=np.int32)
 
    if shuffl==True:
        X, y = shuffle(X, y, random_state=seed)
    return X, y

def load_scores():
    # loading data
    s1 = pd.read_excel("alltrial_pipe_std_bbnb_prob_score.xlsx")['average'].values
    s2 = pd.read_excel("alltrial_pipe_std_blsvc_prob_score.xlsx")['average'].values
    s3 = pd.read_excel("alltrial_pipe_std_dt_prob_score.xlsx")['average'].values
    s4 = pd.read_excel("alltrial_pipe_std_lr_prob_score.xlsx")['average'].values
    s5 = pd.read_csv("alltrial_pipe_std_rf_prob_score.csv")['average'].values

    y_true = pd.read_excel('Ys.xlsx')

    # prepare params for dataframe
    tdata = np.array([s1, s2, s3, s4, s5]).T
    cols = ['bbnb', 'blsvc', 'dt', 'lr', 'rf']

    df = pd.DataFrame(data=tdata, columns=cols)
    return df, y_true

def print_score(clf, X, y, threshold):
    y_pred = cutoff_predict(clf, X, threshold)
    tr_metrics = compute_measure(y, y_pred)
    print("\naccuracy, sen, spec, ppr, npr")
    print("{}\n".format(tr_metrics))

def pred_feats(estimators, cols_lst):

    print('===============================================')
    # model, best_params, feats = eval_(model, params, X, y, gs_refit)

    feat_importance = np.mean([pipe.named_steps['classifier'].coef_ for pipe in estimators], axis=0)

    importance = pd.DataFrame(columns=['Predictive_feat','Importance'], data={'Predictive_feat' : cols_lst, 'Importance': feat_importance.reshape(-1,)})
                                       
    importance = importance[importance['Importance'] > 0].sort_values(by=['Importance'], ascending=False)

    print('Predictive features (+):')
    print(importance)
    print('===============================================')
    return importance

def pred1_feats(clfcoef, cols_lst):

    print('===============================================')
    # model, best_params, feats = eval_(model, params, X, y, gs_refit)
    importance = pd.DataFrame(columns=['Predictive_feat','Importance'], data={'Predictive_feat' : cols_lst, 'Importance': clfcoef.reshape(-1,)})
    """
    importance = pd.DataFrame(index=[cols_lst], columns=['Predictive_feat','Importance'],
                                data={'Importance': clfcoef.reshape(-1,)})
    """
    importance = importance[importance['Importance'] > 0].sort_values(by=['Importance'], ascending=False)

    print('Predictive features (+):')
    print(importance)
    print('===============================================')
    return importance

def main():
    # parse arguments
    parser = check_parser()
    args = parser.parse_args()

    # define number of trials
    NUM_TRIALS = 5
    if args.numtrial != None:
        NUM_TRIALS = args.numtrial

    # define scalers
    mms = MinMaxScaler()
    stds = StandardScaler()

    # resample data 0 (no sampling), 1 (SMOTE), 2 (pandas.sample)
    res = args.resample

    # define pipelines using a specific algorithm (args.algr)
    smt = SMOTE(random_state=42)

    algr = switch_algorithm(args.algr)
    
    # for neural network and BernoulliNB, it's better to normalize data to [0,1]
    scaler = ('stds', stds)
    if args.algr in [7,8]:
        scaler = ('mms', mms)

    # create pipeline of sampling (optional), scaler, and estimator
    pipe_std = Pipeline([scaler, algr])
    pipe_smt_scaler = Pipeline([('smt', smt), scaler, algr])
    
    models = []
    print(res)
    if res==1:
        models.append(('pipe_smt_scaler_'+algr[0], pipe_smt_scaler))
    else:
        models.append(('pipe_std_'+algr[0], pipe_std))
    print(models)

    # Set up possible values of parameters to optimize over (args.algr)
    p_grid =  switch_paramsGrid(args.algr)
    print(p_grid)

    # Load the datasets
    X, y = load_data(args.seed, True)
    #X, y = load_scores()

    # define number of k-folds for cross validation (default=10)
    k_folds = 10
    if args.kfolds != None:
        k_folds=args.kfolds

    # define threshold for prediction proba (Class 1 >= threshold, otherwise Class 0)
    threshold = 0.5
    if args.threshold != None:
        threshold=args.threshold

    # nested cross validation (gridsearchCV inside kfolds-CV)
    all_trial_results = []
    all_trail_proba_results =[]


    imp_feats_df = pd.DataFrame(columns=['Predictive_feat','Importance'])

    for i in range(NUM_TRIALS):
        # to store results for each fold of 10-fold CV (in each trial)
        results = []
        proba_results = []
        idx_results = []
        y_results = []
        
        # to store results for each  each trial (10-fold CV)
        cv_results = []
        cv_proba_results = []
        cv_idx_results = []
        cv_y_results = []

        print("=============== %s-trial ==================" %(i+1))

        # X, y = shuffle(Xo, yo, random_state=i+42)

        # Choose cross-validation techniques for the inner and outer loops, independently of the dataset.
        # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
        inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=i)
        #inner_cv = KFold(n_splits=k_folds, shuffle=True, random_state=i)
        outer_cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=i)

        # Nested CV with parameter optimization
        fold=0
        for name, model in models:
            for train_idx, test_idx in outer_cv.split(X, y):
                # print fold info
                print('=========== Fold', fold+1, 'out of', k_folds, "===========")
                fold +=1
                
                # get train test sets
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]      
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # resample if res=2
                if res==2: X_train, y_train = resampling(X_train, y_train)
                
                # reshape column vector 
                y_train = y_train.values.reshape(-1,)
                y_test = y_test.values.reshape(-1,)
                
                # define scoring and refit criteries: default 0 ('roc_auc'), 1 ('avg_prec_w'), 2 ('pr_auc')
                scoring, gs_refit = scoring_refit_gridseachCV()
                clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv,scoring=scoring, refit=gs_refit, n_jobs=1, iid=False)
                clf.fit(X_train,y_train)
                
                # scores
                print("Best gridsearch score (roc_auc): ", clf.best_score_)
                print(clf.best_params_)
                print()

                print("Train scores:")
                print_score(clf, X_train, y_train, threshold)

                
                clf_name = algr[0]
                if clf_name in ['dt', 'rf', 'xtr', 'gbt', 'xgbt', 'adb', 'lgbm', 'catb' ]:
                    importance=pred1_feats(clf.best_estimator_.named_steps[algr[0]].feature_importances_, X.columns)
                elif clf_name in ['bxgbt', 'bgbt']:
                    importance=pred1_feats(clf.best_estimator_.named_steps[algr[0]].estimators_.feature_importances_, X.columns)
                elif clf_name in ['bsvc', 'blsvc', 'absvc']:
                    importance=pred_feats(clf.best_estimator_.named_steps[algr[0]].estimators_, X.columns)
                elif clf_name in ['rbfsvc', 'sigmsvc', 'polysvc']:
                    importance=pred_feats(clf.best_estimator_.named_steps[algr[0]], X.columns)
                else:
                    importance=pred1_feats(clf.best_estimator_.named_steps[algr[0]].coef_, X.columns)

                # append importance to imp_feats_df                    
                
                imp_feats_df=imp_feats_df.append(importance)
                
                
                #print(classification_report(y_train, clf.predict(X_train)))

                print("Test scores:")
                print_score(clf, X_test, y_test, threshold)
                #print(classification_report(y_test, clf.predict(X_test)))

                # append scores of each run, y_pred, y_test_proba, test_idx
                y_pred = cutoff_predict(clf, X_test, threshold)
                y_test_proba = cutoff_predict_proba(clf, X_test)

                idx_results.append(test_idx)
                proba_results.append(y_test_proba)
                y_results.append(y_test)


                metrics = compute_measure(y_test, y_pred)
                results.append(metrics)
                
            # print results dataframe
            results=pd.DataFrame(results, index=np.arange(1,len(results)+1))
            results.columns = ['accuracy','sensitivity','specificity','PPR','NPR']
            cv_results.append((name, results))


                
            # flatten idx_results, proba_results, y_results
            idx_flatten = [idx for idx_sublist in idx_results for idx in idx_sublist]
            proba_flatten = [idx for idx_sublist in proba_results for idx in idx_sublist]
            y_flatten = [idx for idx_sublist in y_results for idx in idx_sublist]

            
            proba_results_flatten = [x for _, x in sorted(zip(idx_flatten, proba_flatten))]
            y_results_flatten = [x for _, x in sorted(zip(idx_flatten, y_flatten))]
            

            proba_results_df = pd.DataFrame({'prob' : proba_results_flatten, 
                                             'y' : y_results_flatten, 
                                             'y_t' : y.values.reshape(-1,)},
                                             index=sorted(idx_flatten))
            
            #proba_results_df.to_csv(str(i+1)+'_trial_'+str(name)+'_prob_score.csv', encoding='utf-8')

            all_trail_proba_results.append(proba_results_flatten) 
            

        # print results of 10-fold CV and its average
        for i, j in cv_results:
            print("\n\n=================  %s  =================\n"%i)
            jcp = j.copy()
            jcp.loc['mean'] = jcp.mean()
            print(jcp)
    
    # results of all trials
    all_trail_proba_results = pd.DataFrame(np.transpose(np.array(all_trail_proba_results)), index=range(1,len(y)+1), columns=range(1,NUM_TRIALS+1))
    all_trail_proba_results['average'] = all_trail_proba_results.mean(axis=1)
    all_trail_proba_results['true_labels'] = y.values.reshape(-1,)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    all_trail_proba_results.to_csv(timestamp+'_alltrial_'+str(name)+'_prob_score.csv', encoding='utf-8')
    
    #imp_feats_df.to_csv(timestamp+'_impfeats_'+str(name)+'.csv', encoding='utf-8')
    dfr = imp_feats_df.groupby(['Predictive_feat']).sum().sort_values(by=['Importance'], ascending=False)
    dfr['Importance'] /= (k_folds * NUM_TRIALS)
    dfr.to_csv(timestamp+'_impfeats_overall_'+str(name)+'.csv', encoding='utf-8')
    
    
    replace_dict = {r'_0$':'', r'_1$':'', r'_2$':'', r'_3$':'', r'_4$':'', r'_6$':'', r'_12$':'', r'_18$':'', r'_24$':''}
    if clf_name in ['bsvc', 'blsvc', 'lr', 'dt', 'rf', 'xgbt', 'lgbm', 'catb' ]:
        postfix_dict = {
            r'_.1$':'',
            r'.1$':'',
            r'_Q$':'',
            r'_-1$':'',
            r'_Q$':'',
            r'_F$':'',
            r'_M$':'',
            r'_Y$':'',
            r'_N$':'',
            r'_White$':'',
            r'_Asian$':'', r'_Unknown or not reported':'', r'_Not hispanic or latino':'', r'_Hispanic or latino':''
        }
        dft = imp_feats_df.copy()
        dft.replace(to_replace=postfix_dict, value=None, regex=True, inplace=True)
        dft.replace(to_replace=replace_dict, value=None, regex=True, inplace=True)
        dfc = dft.groupby(['Predictive_feat']).sum().sort_values(by=['Importance'], ascending=False)
    else:
        dfc = imp_feats_df.replace(to_replace=replace_dict, value=None, regex=True).groupby(['Predictive_feat']).sum().sort_values(by=['Importance'], ascending=False)
    # scaler Importance scores, so we can do ensemble for predictive features
    dfc['Importance'] /= (k_folds * NUM_TRIALS)
    dfc.to_csv(timestamp+'_combine_impfeats_overall_'+str(name)+'.csv', encoding='utf-8')
    
    
    
    # results after all trials
    print("After %s trials" %NUM_TRIALS)
    print(classification_report(all_trail_proba_results['true_labels'], 
                                ((all_trail_proba_results['average']>=threshold).astype(int))))



    # output results with different thresholds

    y_true = all_trail_proba_results['true_labels']
    y_probs = all_trail_proba_results['average']
    

    print('Test auc score %s' %roc_auc_func(y_true, y_probs))

    fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)


    for class_1_min_recall in [0.5,	0.51,	0.52,	0.53,	0.54,	0.55,	0.56,	0.57,	0.58	,0.59	,0.6,	0.61,	0.62,	0.63,	0.64,	0.65,	0.66,	0.67	,0.68,	0.69,	0.7,	0.71,	0.72,	0.73	,0.74	,0.75,	0.76,	0.77,	0.78,	0.79,	0.8,	0.81,	0.82,	0.83,	0.84,	0.85,	0.86,	0.87,	0.88,	0.89,	0.9,	0.91,	0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99
]:
        print("=============================")
        print("Sens >= %s :" % class_1_min_recall)
        print_res(fpr, tpr, thresholds, class_1_min_recall)
        print()


if __name__ == "__main__":
    start = time.time()
    main()
    print('Time='+str((time.time()-start)))
