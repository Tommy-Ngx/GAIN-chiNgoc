'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.tree import  DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from tqdm import tqdm
import xlwt
from sklearn.model_selection import train_test_split
from utils import normalization
from data_loader import data_loader, make_missing_data
# from autoimpute.imputations import MiceImputer
# from missingpy import MissForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
def main ():
    # data_names = ['spam', 'letter','breast','credit','balance','banknote','blood',
    #               'connectionistvowel','vehicle','yeast']
    # data_names = ['spam']
    # data_names = ['letter']
    # data_names = ['balance','banknote','blood','breasttissue', 'climate','connectionistvowel',
    #               'ecoli','glass','hillvalley','ionosphere', 'parkinsons','planning','seedst',
    #               'thyroid','vehicle','vertebral','wine','yeast']
    # data_names = ['spam', 'letter','breast','banknote']
    data_names = ['parkinsons']
    miss_rate = 0.1
    batch_size = 128
    alpha = 100
    iterations = 10000
    n_times = 30

    wb = xlwt.Workbook()
    sh_dct_mice = wb.add_sheet("DCT_mice")
    sh_mlp_mice = wb.add_sheet("MLP_mice")
    sh_dct_miss = wb.add_sheet("DCT_miss")
    sh_mlp_miss = wb.add_sheet("MLP_miss")

    for k in range(len(data_names)):
        data_name = data_names[k]
        sh_dct_mice.write(0, k, data_name)
        sh_mlp_mice.write(0, k, data_name)
        sh_dct_miss.write(0, k, data_name)
        sh_mlp_miss.write(0, k, data_name)
        print("Dataset: ", data_name)
        ori_data_x, y = data_loader(data_name)
        train_idx, test_idx = train_test_split(range(len(y)), test_size=0.3, stratify=y, random_state=42)
        for i in tqdm(range(n_times)):
            miss_data_x, m = make_missing_data(ori_data_x, miss_rate, seed=i)
            # ###########################MICE imputation#################################

            imp = IterativeImputer(estimator = DecisionTreeRegressor(), max_iter = 20)
            imputed_data_x = imp.fit_transform(miss_data_x)

            # # Normalize data and classification
            imputed_data_x, _ = normalization(imputed_data_x)


            clf = MLPClassifier(hidden_layer_sizes=len(imputed_data_x[1])//2, max_iter=500,
                                early_stopping=True, learning_rate='constant', learning_rate_init=0.1)
            clf.fit(imputed_data_x[train_idx], y[train_idx])
            score = clf.score(imputed_data_x[test_idx], y[test_idx])
            sh_mlp_mice.write(i+1, k, np.round(score,4))

            clf = DecisionTreeClassifier()
            clf.fit(imputed_data_x[train_idx], y[train_idx])
            score = clf.score(imputed_data_x[test_idx], y[test_idx])
            sh_dct_mice.write(i+1, k, np.round(score,4))

            # ###########################Miss imputation#################################

            imp = IterativeImputer(max_iter = 20)
            imputed_data_x = imp.fit_transform(miss_data_x)
            imputed_data_x, _ = normalization(imputed_data_x)
            train_idx, test_idx = train_test_split(range(len(y)), test_size=0.3, stratify=y, random_state=42)

            clf = MLPClassifier(hidden_layer_sizes=len(imputed_data_x[1])//2, max_iter=500,
                                early_stopping=True, learning_rate='constant', learning_rate_init=0.1)
            clf.fit(imputed_data_x[train_idx], y[train_idx])
            score = clf.score(imputed_data_x[test_idx], y[test_idx])
            sh_mlp_miss.write(i + 1, k, np.round(score, 4))

            clf = DecisionTreeClassifier()
            clf.fit(imputed_data_x[train_idx], y[train_idx])
            score = clf.score(imputed_data_x[test_idx], y[test_idx])
            sh_dct_miss.write(i + 1, k, np.round(score, 4))
    wb.save("MICE_MISS_10_parkinsons.xls")
if __name__ == '__main__':
    main()
