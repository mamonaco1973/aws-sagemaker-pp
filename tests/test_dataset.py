import numpy as np
import pandas as pd
import io

import dataset


def test_same_seed_same_rows_every_time():
    a_x, a_y = dataset.generate()
    b_x, b_y = dataset.generate()
    assert np.array_equal(a_x, b_x) and a_y == b_y


def test_new_readings_are_not_the_training_rows():
    train_x, _ = dataset.generate()
    new_x, _ = dataset.generate(n_rows=200, seed=dataset.NEW_READINGS_SEED)
    train_rows = {tuple(r) for r in train_x}
    assert not any(tuple(r) in train_rows for r in new_x)


def test_labels_are_probabilistic_not_a_clean_rule():
    rng = np.random.default_rng(dataset.TRAINING_SEED)
    t = rng.normal(70.0, 8.0, 5000)
    v = rng.lognormal(np.log(2.8), 0.45, 5000)
    h = rng.uniform(0.0, 12000.0, 5000)
    p = dataset.failure_probability(t, v, h)
    # A large share of machines sit in the uncertain middle, where the coin
    # flip decides, so no model can score perfectly.
    assert np.mean((p > 0.2) & (p < 0.8)) > 0.15
    _, labels = dataset.generate()
    rate = np.mean(np.array(labels) == "failure")
    assert 0.15 < rate < 0.40, "failures should be a clear minority"


def test_split_is_stratified_and_disjoint():
    x, y = dataset.generate()
    x_train, x_test, y_train, y_test = dataset.split(x, y)
    assert len(x_train) == 4000 and len(x_test) == 1000
    train_rate = np.mean(np.array(y_train) == "failure")
    test_rate = np.mean(np.array(y_test) == "failure")
    assert abs(train_rate - test_rate) < 0.005
    train_rows = {tuple(r) for r in x_train}
    assert sum(tuple(r) in train_rows for r in x_test) == 0


def test_csv_has_named_columns_in_feature_order():
    x, y = dataset.generate(n_rows=10)
    frame = pd.read_csv(io.StringIO(dataset.to_csv(x, y)))
    assert list(frame.columns) == list(dataset.FEATURES) + [dataset.TARGET]
    assert set(frame[dataset.TARGET]) <= set(dataset.LABELS)
    np.testing.assert_allclose(frame[list(dataset.FEATURES)].to_numpy(), x, atol=0.006)
