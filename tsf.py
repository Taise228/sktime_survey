from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head

classifier = TimeSeriesForestClassifier(n_estimators=100)
arrow_train_X_2d, arrow_train_y_2d = load_arrow_head(
    split="train", return_type="numpy2d"
)
arrow_test_X_2d, arrow_test_y_2d = load_arrow_head(split="test", return_type="numpy2d")
classifier.fit(arrow_train_X_2d, arrow_train_y_2d)

print(classifier.feature_importances_.shape, arrow_train_X_2d.shape)
print(classifier.estimators_[0].feature_importances_)