import time

import numpy as np
from sktime.datasets import load_basic_motions
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.hybrid import HIVECOTEV2


def main(model):
    # motion_x, motion_y = load_basic_motions(return_type='numpy3d')
    # motion_x.shape = (80, 6, 100) -> 80サンプル、それぞれ6点 x 100 time point
    # instance_0 = motion_x[0] -> 最初のサンプル
    # instance_0 の0 time point でのデータは [instance_0[0][0], instance_0[1][0], ..., instance_0[5][0]]

    motion_train_x, motion_train_y = load_basic_motions(split='train', return_type='numpy3d')
    motion_test_x, motion_test_y = load_basic_motions(split='test', return_type='numpy3d')

    print(motion_train_x.shape, motion_train_y.shape, motion_test_x.shape, motion_test_y.shape)

    start = time.time()
    if model == 'rocket':
        rocket = RocketClassifier()
        print('model downloaded')
        rocket.fit(motion_train_x, motion_train_y)
        print('training complete')
        print('time elapsed:', time.time() - start)
        y_pred_rocket = rocket.predict(motion_test_x)
        accuracy_rocket = np.sum(y_pred_rocket == motion_test_y) / y_pred_rocket.shape[0]
        print(accuracy_rocket)
    elif model == 'hc2':
        hc2 = HIVECOTEV2(time_limit_in_minutes=1)
        print('model downloaded')
        hc2.fit(motion_train_x, motion_train_y)
        print('training complete')
        print('time elapsed:', time.time() - start)
        y_pred_hc2 = hc2.predict(motion_test_x)
        accuracy_hc2 = np.sum(y_pred_hc2 == motion_test_y) / y_pred_hc2.shape[0]
        print(accuracy_hc2)


if __name__ == '__main__':
    main('hc2')
