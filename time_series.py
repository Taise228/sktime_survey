import time

import numpy as np
from sktime.datasets import load_basic_motions
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.early_classification import TEASER
from sktime.classification.deep_learning import CNNClassifier
from sktime.classification.shapelet_based import MrSQM
from sktime.classification.interval_based import TimeSeriesForestClassifier, DrCIF
from pyts.classification import TimeSeriesForest
from pyts.multivariate.classification import MultivariateClassifier


def main(model_name):
    # motion_x, motion_y = load_basic_motions(return_type='numpy3d')
    # motion_x.shape = (80, 6, 100) -> 80サンプル、それぞれ6点 x 100 time point
    # instance_0 = motion_x[0] -> 最初のサンプル
    # instance_0 の0 time point でのデータは [instance_0[0][0], instance_0[1][0], ..., instance_0[5][0]]

    motion_train_x, motion_train_y = load_basic_motions(split='train', return_type='numpy3d')
    motion_test_x, motion_test_y = load_basic_motions(split='test', return_type='numpy3d')

    print(motion_train_x.shape, motion_train_y.shape, motion_test_x.shape, motion_test_y.shape)

    start = time.time()
    if model_name == 'rocket':
        model = RocketClassifier()
    elif model_name == 'hc2':
        model = HIVECOTEV2(time_limit_in_minutes=1)
    elif model_name == 'teaser':
        model = TEASER()
    elif model_name == 'pyts':
        model = MultivariateClassifier(TimeSeriesForest(class_weight='balanced'))
    elif model_name == 'cnn':
        model = CNNClassifier(n_epochs=20, batch_size=4)
    elif model_name == 'mrsqm':
        model = MrSQM()
    elif model_name == 'forest':
        model = MultivariateClassifier(TimeSeriesForestClassifier())
    elif model_name == 'drcif':
        model = DrCIF(n_estimators=200, n_intervals=5)

    print('model downloaded')
    model.fit(motion_train_x, motion_train_y)
    print('training complete')
    print('time elapsed for training:', time.time() - start)
    predict_start = time.time()
    y_pred = model.predict(motion_test_x)
    print('time elapsed for prediction', time.time() - predict_start)
    accuracy = np.sum(y_pred == motion_test_y) / y_pred.shape[0]
    print(accuracy)
    # print(type(model.estimator))


if __name__ == '__main__':
    main('drcif')
