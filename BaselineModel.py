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
from sklearn.neural_network import MLPClassifier
def main ():
    # data_names = ['spam', 'letter','breast','credit','balance','banknote','blood',
    #               'connectionistvowel','vehicle','yeast']
    # data_names = ['spam']
    # data_names = ['letter']
    # data_names = ['balance','banknote','blood','breasttissue', 'climate','connectionistvowel',
    #               'ecoli','glass','hillvalley','ionosphere', 'parkinsons','planning','seedst',
    #               'thyroid','vehicle','vertebral','wine','yeast']
    data_names = ['spam', 'letter','breast','banknote']
    # data_names = ['parkinsons']
    miss_rate = 0.1
    batch_size = 128
    alpha = 100
    iterations = 10000
    n_times = 30

    wb = xlwt.Workbook()
    sh_dct_mean = wb.add_sheet("DCT_mean")
    sh_mlp_mean = wb.add_sheet("MLP_mean")
    sh_dct_knn = wb.add_sheet("DCT_knn")
    sh_mlp_knn = wb.add_sheet("MLP_knn")

    for k in range(len(data_names)):
        data_name = data_names[k]
        sh_dct_mean.write(0, k, data_name)
        sh_mlp_mean.write(0, k, data_name)
        sh_dct_knn.write(0, k, data_name)
        sh_mlp_knn.write(0, k, data_name)
        print("Dataset: ", data_name)
        ori_data_x, y = data_loader(data_name)
        train_idx, test_idx = train_test_split(range(len(y)), test_size=0.3, stratify=y, random_state=42)
        for i in tqdm(range(n_times)):
            miss_data_x, m = make_missing_data(ori_data_x, miss_rate, seed=i)
            # ###########################Mean imputation#################################

            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputed_data_x = imp.fit_transform(miss_data_x)

            # # Normalize data and classification
            imputed_data_x, _ = normalization(imputed_data_x)


            clf = MLPClassifier(hidden_layer_sizes=len(imputed_data_x[1])//2, max_iter=500,
                                early_stopping=True, learning_rate='constant', learning_rate_init=0.01)
            clf.fit(imputed_data_x[train_idx], y[train_idx])
            score = clf.score(imputed_data_x[test_idx], y[test_idx])
            sh_mlp_mean.write(i+1, k, np.round(score,4))

            clf = DecisionTreeClassifier()
            clf.fit(imputed_data_x[train_idx], y[train_idx])
            score = clf.score(imputed_data_x[test_idx], y[test_idx])
            sh_dct_mean.write(i+1, k, np.round(score,4))

            # ###########################KNN imputation#################################

            imp = KNNImputer(missing_values=np.nan)
            imputed_data_x = imp.fit_transform(miss_data_x)
            imputed_data_x, _ = normalization(imputed_data_x)
            train_idx, test_idx = train_test_split(range(len(y)), test_size=0.3, stratify=y, random_state=42)

            clf = MLPClassifier(hidden_layer_sizes=len(imputed_data_x[1])//2, max_iter=500,
                                early_stopping=True, learning_rate='constant', learning_rate_init=0.01)
            clf.fit(imputed_data_x[train_idx], y[train_idx])
            score = clf.score(imputed_data_x[test_idx], y[test_idx])
            sh_mlp_knn.write(i + 1, k, np.round(score, 4))

            clf = DecisionTreeClassifier()
            clf.fit(imputed_data_x[train_idx], y[train_idx])
            score = clf.score(imputed_data_x[test_idx], y[test_idx])
            sh_dct_knn.write(i + 1, k, np.round(score, 4))
    wb.save("Baseline_10.xls")
if __name__ == '__main__':
    main()
