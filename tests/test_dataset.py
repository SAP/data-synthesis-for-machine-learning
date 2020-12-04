
from pandas import DataFrame
from numpy import array_equal

from ds4ml.dataset import DataSet

from .testdata import adults01


def test_encode():
    from .testdata import adults01
    from numpy import array_equal
    dataset = DataSet(adults01)
    frame = dataset.encode()
    for col in ['education', 'relationship', 'salary']:
        assert col not in frame.columns
    for col in ['age', 'birth']:
        assert col in frame.columns

    assert 'salary_<=50K' in frame.columns
    assert 'salary_>50K' in frame.columns

    for attr, val in [('salary', '<=50K'),
                      ('relationship', 'Wife'),
                      ('relationship', 'Husband')]:
        trans_col = frame[f'{attr}_{val}'].apply(lambda v: v == 1)
        origin_col = adults01[attr] == val
        assert array_equal(trans_col, origin_col)


def test_encode_partly():
    from .testdata import adults01
    from sklearn.model_selection import train_test_split
    dataset = DataSet(adults01)
    train, test = train_test_split(adults01, test_size=0.2)
    frame = dataset.encode(data=train)
    assert 'salary_<=50K' in frame.columns
    assert 'salary_>50K' in frame.columns
    assert ((0 == frame['salary_<=50K']) | (frame['salary_<=50K'] == 1)).all()
    assert ((0.0 <= frame['age']) & (frame['age'] <= 1.0)).all()


def test_encode_empty_column():
    from numpy import array_equal
    data = [[1001, 'A', 'Female'],
            [1002, 'B', 'Male'],
            [1003, 'C', 'Male'],
            [1004, 'D', 'Female'],
            [1005, 'E', 'Female']]
    ds = DataSet(data, columns=['ID', 'Name', 'Sex'])
    x = DataFrame(data[-2:], columns=['ID', 'Name', 'Sex'])
    x_tf = ds.encode(data=x)
    # Name is not categorical, because it has unique values
    assert x_tf.shape == (2, 3)
    assert array_equal(x_tf.columns, ['ID', 'Sex_Female', 'Sex_Male'])


def test_svm_task():
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from .testdata import adults01
    c_df = DataFrame(adults01)
    c_tf = DataSet(c_df).encode()
    train, test = train_test_split(c_tf, test_size=0.2)

    def make_train_x_y(df):
        x_ = df.drop(['salary_<=50K', 'salary_>50K'], axis=1)
        # <=50K and >50K are binary, complementary
        _, ym_ = df['salary_<=50K'], df['salary_>50K']
        return x_, ym_

    tr_x, tr_y = make_train_x_y(train)
    te_x, te_y = make_train_x_y(test)
    clf = SVC(gamma='scale')
    clf.fit(tr_x, tr_y)
    pr_y = clf.predict(te_x)
    from sklearn.metrics import confusion_matrix, classification_report
    print(confusion_matrix(te_y, pr_y))
    print(classification_report(te_y, pr_y))


def test_synthesize():
    dataset = DataSet(adults01)
    df = dataset.synthesize()
    assert df.size == dataset.size


def test_synthesize_with_pseudonyms():
    dataset = DataSet(adults01)
    df = dataset.synthesize(pseudonyms=['salary'])
    assert df.size == dataset.size
    assert array_equal(dataset['salary'].value_counts().values,
                       df['salary'].value_counts().values)


def test_synthesize_with_retains():
    dataset = DataSet(adults01)
    df = dataset.synthesize(retains=['age'])
    assert df.size == dataset.size
    assert array_equal(dataset['age'], df['age'])


def test_synthesize_for_privacy():
    # Verify probability after synthesis by differential privacy. (This test
    # case may fail because of limit runs.)
    from numpy.random import randint
    from numpy import exp
    epsilon = 0.1
    runs = 200
    data = randint(65, 90, size=(199, 2))
    set1 = DataSet(data.tolist() + [[65, 65]], columns=['ColA', 'ColB'])
    set2 = DataSet(data.tolist() + [[65, 66]], columns=['ColA', 'ColB'])
    counts = [0, 0]
    for i in range(runs):
        df1 = set1.synthesize(epsilon=epsilon)
        df2 = set2.synthesize(epsilon=epsilon)
        counts[0] += ((df1['ColA'] == 65) & (df1['ColB'] == 65)).sum()
        counts[1] += ((df2['ColA'] == 65) & (df2['ColB'] == 66)).sum()
    assert counts[0] / (runs * 200) <= exp(epsilon) * counts[1] / (runs * 200)
