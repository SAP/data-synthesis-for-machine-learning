"""
Test Data is from part of https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
"""
from pandas import DataFrame

adults01 = DataFrame([
    [39, 'Bachelors', 'Not-in-family', '<=50K', '06/26/1980'],
    [50, 'Bachelors', 'Husband', '<=50K', '06/18/1969'],
    [38, 'HS-grad', 'Not-in-family', '<=50K', '06/27/1981'],
    [53, '11th', 'Husband', '<=50K', '06/16/1966'],
    [28, 'Bachelors', 'Wife', '<=50K', '07/05/1991'],
    [37, 'Masters', 'Wife', '<=50K', '06/28/1982'],
    [49, '9th', 'Not-in-family', '<=50K', '06/19/1970'],
    [52, 'HS-grad', 'Husband', '>50K', '06/17/1967'],
    [31, 'Masters', 'Not-in-family', '>50K', '07/02/1988'],
    [42, 'Bachelors', 'Husband', '>50K', '06/24/1977'],
    [37, 'Some-college', 'Husband', '>50K', '06/28/1982'],
    [30, 'Bachelors', 'Husband', '>50K', '07/03/1989'],
    [23, 'Bachelors', 'Own-child', '<=50K', '07/08/1996'],
    [32, 'Bachelors', 'Not-in-family', '<=50K', '07/02/1987'],
    [34, '7th-8th', 'Husband', '<=50K', '06/30/1985'],
    [25, 'HS-grad', 'Own-child', '<=50K', '07/07/1994'],
    [32, 'HS-grad', 'Unmarried', '<=50K', '07/02/1987'],
    [38, '11th', 'Husband', '<=50K', '06/27/1981'],
    [43, 'Masters', 'Unmarried', '>50K', '06/23/1976'],
    [40, 'Doctorate', 'Husband', '>50K', '06/26/1979'],
    [54, 'HS-grad', 'Unmarried', '<=50K', '06/15/1965'],
    [35, '9th', 'Husband', '<=50K', '06/29/1984'],
    [43, '11th', 'Husband', '<=50K', '06/23/1976'],
    [59, 'HS-grad', 'Unmarried', '<=50K', '06/11/1960'],
    [56, 'Bachelors', 'Husband', '>50K', '06/14/1963'],
    [19, 'HS-grad', 'Own-child', '<=50K', '07/11/2000'],
    [39, 'HS-grad', 'Not-in-family', '<=50K', '06/26/1980'],
    [49, 'HS-grad', 'Husband', '<=50K', '06/19/1970'],
    [23, 'Assoc-acdm', 'Not-in-family', '<=50K', '07/08/1996'],
    [20, 'Some-college', 'Own-child', '<=50K', '07/11/1999']
], columns=['age', 'education', 'relationship', 'salary', 'birth'])


adults02 = DataFrame([
    [19, 'HS-grad', 'Own-child', 'Male', '<=50K', '07/11/2000'],
    [26, 'Bachelors', 'Own-child', 'Male', '<=50K', '07/06/1993'],
    [27, 'Some-college', 'Not-in-family', 'Male', '<=50K', '07/05/1992'],
    [41, 'Masters', 'Husband', 'Male', '<=50K', '06/25/1978'],
    [33, 'Doctorate', 'Husband', 'Male', '<=50K', '07/01/1986'],
    [56, 'Some-college', 'Not-in-family', 'Male', '<=50K', '06/14/1963'],
    [43, 'Bachelors', 'Husband', 'Male', '>50K', '06/23/1976'],
    [29, 'HS-grad', 'Wife', 'Female', '<=50K', '07/04/1990'],
    [44, '11th', 'Husband', 'Male', '>50K', '06/23/1975'],
    [37, 'Some-college', 'Own-child', 'Female', '<=50K', '06/28/1982'],
    [24, 'Some-college', 'Not-in-family', 'Male', '<=50K', '07/08/1995'],
    [38, 'HS-grad', 'Husband', 'Male', '<=50K', '06/27/1981'],
    [35, 'Masters', 'Husband', 'Male', '>50K', '06/29/1984'],
    [39, 'Bachelors', 'Own-child', 'Female', '<=50K', '06/26/1980'],
    [47, 'HS-grad', 'Husband', 'Male', '>50K', '06/20/1972'],
    [51, 'HS-grad', 'Husband', 'Male', '>50K', '06/17/1968'],
    [38, 'HS-grad', 'Husband', 'Male', '<=50K', '06/27/1981'],
    [44, 'Some-college', 'Unmarried', 'Female', '<=50K', '06/23/1975'],
    [24, 'HS-grad', 'Other-relative', 'Female', '<=50K', '07/08/1995'],
    [41, 'HS-grad', 'Unmarried', 'Female', '<=50K', '06/25/1978'],
    [51, 'Assoc-voc', 'Unmarried', 'Female', '<=50K', '06/17/1968'],
    [60, 'HS-grad', 'Husband', 'Male', '<=50K', '06/11/1959'],
    [40, 'Bachelors', 'Husband', 'Male', '>50K', '06/26/1979'],
    [27, 'Some-college', 'Wife', 'Female', '<=50K', '07/05/1992'],
    [36, 'HS-grad', 'Husband', 'Male', '>50K', '06/29/1983'],
    [44, 'HS-grad', 'Husband', 'Male', '<=50K', '06/23/1975'],
    [33, 'Some-college', None, 'Female', '<=50K', '07/01/1986'],
    [53, '7th-8th', 'Husband', 'Male', '<=50K', '06/16/1966'],
    [43, 'HS-grad', 'Husband', 'Male', '>50K', '06/23/1976'],
    [44, 'Assoc-acdm', 'Not-in-family', 'Male', '<=50K', '06/23/1975'],
], columns=['age', 'education', 'relationship', 'sex', 'salary', 'birth'])

adult_with_head = 'age, education, relationship, sex, salary, birth\n' \
           '19, HS-grad, Own-child, Male, <=50K, 07/11/2000\n' \
           '41, Masters, Husband, Male, <=50K, 06/25/1978\n' \
           '44, HS-grad, Husband, Male, <=50K, 06/23/1975'


adult_without_head = '19, HS-grad, Own-child, Male, <=50K, 07/11/2000\n' \
           '41, Masters, Husband, Male, <=50K, 06/25/1978\n' \
           '40, Masters, Husband, Female, <=50K, 06/21/1977\n' \
           '44, HS-grad, Husband, Male, <=50K, 06/23/1975'


adult_with_head_res = DataFrame([
    [19, 'HS-grad', 'Own-child', 'Male', '<=50K', '07/11/2000'],
    [41, 'Masters', 'Husband', 'Male', '<=50K', '06/25/1978'],
    [44, 'HS-grad', 'Husband', 'Male', '<=50K', '06/23/1975']
], columns=['age', 'education', 'relationship', 'sex', 'salary', 'birth'])