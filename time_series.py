from sktime.datasets import load_basic_motions


def main():
    # motion_x, motion_y = load_basic_motions(return_type='numpy3d')
    # motion_x.shape = (80, 6, 100) -> 80サンプル、それぞれ6点 x 100 time point
    # instance_0 = motion_x[0] -> 最初のサンプル
    # instance_0 の0 time point でのデータは [instance_0[0][0], instance_0[1][0], ..., instance_0[5][0]]

    motion_train_x, motion_train_y = load_basic_motions(split='train', return_type='numpy3d')
    motion_test_x, motion_test_y = load_basic_motions(split='test', return_type='numpy3d')

    print(motion_train_x.shape, motion_train_y.shape, motion_test_x.shape, motion_test_y.shape)


if __name__ == '__main__':
    main()