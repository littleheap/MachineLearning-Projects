import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("./LoanStats_2016Q3.csv", skiprows=1, low_memory=False)

# 输出数据信息
print(df.info())

'''
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 99122 entries, 0 to 99121
    Columns: 122 entries, id to sec_app_mths_since_last_major_derog
    dtypes: float64(97), object(25)
    memory usage: 92.3+ MB
    None
'''

# 打印前三行
print(df.head(3))  # [3 rows x 122 columns]

'''
        id  member_id  loan_amnt  funded_amnt  funded_amnt_inv        term  \
    0  NaN        NaN    15000.0      15000.0          15000.0   36 months   
    1  NaN        NaN     2600.0       2600.0           2600.0   36 months   
    2  NaN        NaN    32200.0      32200.0          32200.0   60 months   
    
      int_rate  installment grade sub_grade                 ...                  \
    0   13.99%       512.60     C        C3                 ...                   
    1    8.99%        82.67     B        B1                 ...                   
    2   21.49%       880.02     D        D5                 ...                   
    
      sec_app_earliest_cr_line sec_app_inq_last_6mths sec_app_mort_acc  \
    0                      NaN                    NaN              NaN   
    1                      NaN                    NaN              NaN   
    2                      NaN                    NaN              NaN   
    
       sec_app_open_acc sec_app_revol_util sec_app_open_il_6m  \
    0               NaN                NaN                NaN   
    1               NaN                NaN                NaN   
    2               NaN                NaN                NaN   
    
      sec_app_num_rev_accts sec_app_chargeoff_within_12_mths  \
    0                   NaN                              NaN   
    1                   NaN                              NaN   
    2                   NaN                              NaN   
    
       sec_app_collections_12_mths_ex_med sec_app_mths_since_last_major_derog  
    0                                 NaN                                 NaN  
    1                                 NaN                                 NaN  
    2                                 NaN                                 NaN  
'''

# 特征筛选
# .ix[row slice, column slice]
print(df.ix[:4, :7])

'''
        id  member_id  loan_amnt  funded_amnt  funded_amnt_inv        term  \
    0  NaN        NaN    15000.0      15000.0          15000.0   36 months   
    1  NaN        NaN     2600.0       2600.0           2600.0   36 months   
    2  NaN        NaN    32200.0      32200.0          32200.0   60 months   
    3  NaN        NaN    10000.0      10000.0          10000.0   36 months   
    4  NaN        NaN     6000.0       6000.0           6000.0   36 months   
    
      int_rate  
    0   13.99%  
    1    8.99%  
    2   21.49%  
    3   11.49%  
    4   13.49%  
'''

# 删除两列空数据
df.drop('id', 1, inplace=True)
df.drop('member_id', 1, inplace=True)
# 百分号去掉并改变字段类型
df.int_rate = pd.Series(df.int_rate).str.replace('%', '').astype(float)

print(df.ix[:4, :7])
'''
       loan_amnt  funded_amnt  funded_amnt_inv        term  int_rate  installment  \
    0    15000.0      15000.0          15000.0   36 months     13.99       512.60   
    1     2600.0       2600.0           2600.0   36 months      8.99        82.67   
    2    32200.0      32200.0          32200.0   60 months     21.49       880.02   
    3    10000.0      10000.0          10000.0   36 months     11.49       329.72   
    4     6000.0       6000.0           6000.0   36 months     13.49       203.59   
    
      grade  
    0     C  
    1     B  
    2     D  
    3     B  
    4     C  
'''

print((df.loan_amnt != df.funded_amnt).value_counts())
'''
    False    99120
    True         2
    dtype: int64
'''

print(df.query('loan_amnt != funded_amnt').head(5))  # [2 rows x 120 columns]
'''
           loan_amnt  funded_amnt  funded_amnt_inv term  int_rate  installment  \
    99120        NaN          NaN              NaN  NaN       NaN          NaN   
    99121        NaN          NaN              NaN  NaN       NaN          NaN   
    
          grade sub_grade emp_title emp_length  \
    99120   NaN       NaN       NaN        NaN   
    99121   NaN       NaN       NaN        NaN   
    
                          ...                 sec_app_earliest_cr_line  \
    99120                 ...                                      NaN   
    99121                 ...                                      NaN   
    
           sec_app_inq_last_6mths sec_app_mort_acc sec_app_open_acc  \
    99120                     NaN              NaN              NaN   
    99121                     NaN              NaN              NaN   
    
          sec_app_revol_util sec_app_open_il_6m  sec_app_num_rev_accts  \
    99120                NaN                NaN                    NaN   
    99121                NaN                NaN                    NaN   
    
          sec_app_chargeoff_within_12_mths sec_app_collections_12_mths_ex_med  \
    99120                              NaN                                NaN   
    99121                              NaN                                NaN   
    
          sec_app_mths_since_last_major_derog  
    99120                                 NaN  
    99121                                 NaN  
'''

# 删除行空数据
df.dropna(axis=0, how='all', inplace=True)

print(df.info())
'''
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 99120 entries, 0 to 99119
    Columns: 120 entries, loan_amnt to sec_app_mths_since_last_major_derog
    dtypes: float64(97), object(23)
    memory usage: 91.5+ MB
    None
'''

# 删除列空数据
df.dropna(axis=1, how='all', inplace=True)

print(df.info())
'''
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 99120 entries, 0 to 99119
    Columns: 108 entries, loan_amnt to total_il_high_credit_limit
    dtypes: float64(85), object(23)
    memory usage: 82.4+ MB
    None
'''

# 打印一部分数据
print(df.ix[:5, 8:15])
'''
                  emp_title emp_length home_ownership  annual_inc  \
    0       Fiscal Director    2 years           RENT     55000.0   
    1    Loaner Coordinator    3 years           RENT     35000.0   
    2  warehouse/supervisor  10+ years       MORTGAGE     65000.0   
    3               Teacher  10+ years            OWN     55900.0   
    4           SERVICE MGR    5 years           RENT     33000.0   
    5       General Manager  10+ years       MORTGAGE    109000.0   
    
      verification_status   issue_d loan_status  
    0        Not Verified  Sep-2016     Current  
    1     Source Verified  Sep-2016  Fully Paid  
    2        Not Verified  Sep-2016  Fully Paid  
    3        Not Verified  Sep-2016     Current  
    4        Not Verified  Sep-2016     Current  
    5     Source Verified  Sep-2016     Current  
'''

print(df.emp_title.value_counts().head())
'''
    Teacher       1931
    Manager       1701
    Owner          990
    Supervisor     785
    Driver         756
    Driver         756
    Name: emp_title, dtype: int64
'''

print(df.emp_title.value_counts().tail())
'''
     Center Manager              1
    COLLECTIONS                  1
    Training Facility Manager    1
    Owner, Creative Director     1
    Data Management Analyst      1
    Name: emp_title, dtype: int64
'''

print(df.emp_title.unique().shape)  # (37421,)

# 删除emp_title
df.drop(['emp_title'], 1, inplace=True)

print(df.ix[:5, 8:15])
'''
      emp_length home_ownership  annual_inc verification_status   issue_d  \
    0    2 years           RENT     55000.0        Not Verified  Sep-2016   
    1    3 years           RENT     35000.0     Source Verified  Sep-2016   
    2  10+ years       MORTGAGE     65000.0        Not Verified  Sep-2016   
    3  10+ years            OWN     55900.0        Not Verified  Sep-2016   
    4    5 years           RENT     33000.0        Not Verified  Sep-2016   
    5  10+ years       MORTGAGE    109000.0     Source Verified  Sep-2016   
    
      loan_status pymnt_plan  
    0     Current          n  
    1  Fully Paid          n  
    2  Fully Paid          n  
    3     Current          n  
    4     Current          n  
    5     Current          n  
'''

print(df.emp_length.value_counts())
'''
    10+ years    34219
    2 years       9066
    3 years       7925
    < 1 year      7104
    1 year        6991
    5 years       6170
    4 years       6022
    6 years       4406
    8 years       4168
    9 years       3922
    7 years       3205
    Name: emp_length, dtype: int64
'''

# 工作年限规范化
df.replace('n/a', np.nan, inplace=True)
df.emp_length.fillna(value=0, inplace=True)
df['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
df['emp_length'] = df['emp_length'].astype(int)

print(df.emp_length.value_counts())
'''
    10    34219
    1     14095
    2      9066
    3      7925
    5      6170
    4      6022
    0      5922
    6      4406
    8      4168
    9      3922
    7      3205
    Name: emp_length, dtype: int64
'''

# 统计确认信息情况
print(df.verification_status.value_counts())
'''
    Source Verified    40781
    Verified           31356
    Not Verified       26983
    Name: verification_status, dtype: int64
'''

print(df.info())
'''
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 99120 entries, 0 to 99119
    Columns: 107 entries, loan_amnt to total_il_high_credit_limit
    dtypes: float64(85), int32(1), object(21)
    memory usage: 83.8+ MB
    None
'''

print(df.columns)
'''
    Index(['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
           'installment', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
           ...
           'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq',
           'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens',
           'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
           'total_il_high_credit_limit'],
          dtype='object', length=107)
'''

# 打印借款逾期情况
print(pd.unique(df['loan_status'].values.ravel()))
'''
    ['Current' 'Fully Paid' 'Late (31-120 days)' 'Charged Off' 'Late (16-30 days)' 'In Grace Period' 'Default']
'''

for col in df.select_dtypes(include=['object']).columns:
    print("Column {} has {} unique instances".format(col, len(df[col].unique())))
'''
    Column term has 2 unique instances
    Column grade has 7 unique instances
    Column sub_grade has 35 unique instances
    Column home_ownership has 4 unique instances
    Column verification_status has 3 unique instances
    Column issue_d has 3 unique instances
    Column loan_status has 7 unique instances
    Column pymnt_plan has 2 unique instances
    Column desc has 6 unique instances
    Column purpose has 13 unique instances
    Column title has 13 unique instances
    Column zip_code has 873 unique instances
    Column addr_state has 50 unique instances
    Column earliest_cr_line has 614 unique instances
    Column revol_util has 1087 unique instances
    Column initial_list_status has 2 unique instances
    Column last_pymnt_d has 13 unique instances
    Column next_pymnt_d has 4 unique instances
    Column last_credit_pull_d has 14 unique instances
    Column application_type has 3 unique instances
    Column verification_status_joint has 2 unique instances
'''

# 处理对象类型的缺失率
print(df.select_dtypes(include=['O']).describe().T.assign(
    missing_pct=df.apply(lambda x: (len(x) - x.count()) / float(len(x)))))
'''
                               missing_pct  
    term                          0.000000  
    grade                         0.000000  
    sub_grade                     0.000000  
    home_ownership                0.000000  
    verification_status           0.000000  
    issue_d                       0.000000  
    loan_status                   0.000000  
    pymnt_plan                    0.000000  
    desc                          0.999939  
    purpose                       0.000000  
    title                         0.054752  
    zip_code                      0.000000  
    addr_state                    0.000000  
    earliest_cr_line              0.000000  
    revol_util                    0.000605  
    initial_list_status           0.000000  
    last_pymnt_d                  0.001301  
    next_pymnt_d                  0.157062  
    last_credit_pull_d            0.000050  
    application_type              0.000000  
    verification_status_joint     0.994784  
'''

# 删除一些缺失率高和unique高的字段
df.drop('desc', 1, inplace=True)
df.drop('verification_status_joint', 1, inplace=True)
df.drop('zip_code', 1, inplace=True)
df.drop('addr_state', 1, inplace=True)
df.drop('earliest_cr_line', 1, inplace=True)
df.drop('revol_util', 1, inplace=True)
df.drop('purpose', 1, inplace=True)
df.drop('title', 1, inplace=True)
df.drop('term', 1, inplace=True)
df.drop('issue_d', 1, inplace=True)
# df.drop('',1,inplace=True)
# 贷后相关的字段
df.drop(['out_prncp', 'out_prncp_inv', 'total_pymnt',
         'total_pymnt_inv', 'total_rec_prncp', 'grade', 'sub_grade'], 1, inplace=True)
df.drop(['total_rec_int', 'total_rec_late_fee',
         'recoveries', 'collection_recovery_fee',
         'collection_recovery_fee'], 1, inplace=True)
df.drop(['last_pymnt_d', 'last_pymnt_amnt',
         'next_pymnt_d', 'last_credit_pull_d'], 1, inplace=True)
df.drop(['policy_code'], 1, inplace=True)

print(df.info())
'''
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 99120 entries, 0 to 99119
    Data columns (total 81 columns):
    loan_amnt                         99120 non-null float64
    funded_amnt                       99120 non-null float64
    funded_amnt_inv                   99120 non-null float64
    int_rate                          99120 non-null float64
    installment                       99120 non-null float64
    emp_length                        99120 non-null int32
    home_ownership                    99120 non-null object
    annual_inc                        99120 non-null float64
    verification_status               99120 non-null object
    loan_status                       99120 non-null object
    pymnt_plan                        99120 non-null object
    dti                               99120 non-null float64
    delinq_2yrs                       99120 non-null float64
    inq_last_6mths                    99120 non-null float64
    mths_since_last_delinq            53366 non-null float64
    mths_since_last_record            19792 non-null float64
    open_acc                          99120 non-null float64
    pub_rec                           99120 non-null float64
    revol_bal                         99120 non-null float64
    total_acc                         99120 non-null float64
    initial_list_status               99120 non-null object
    collections_12_mths_ex_med        99120 non-null float64
    mths_since_last_major_derog       29372 non-null float64
    application_type                  99120 non-null object
    annual_inc_joint                  517 non-null float64
    dti_joint                         517 non-null float64
    acc_now_delinq                    99120 non-null float64
    tot_coll_amt                      99120 non-null float64
    tot_cur_bal                       99120 non-null float64
    open_acc_6m                       99120 non-null float64
    open_il_6m                        99120 non-null float64
    open_il_12m                       99120 non-null float64
    open_il_24m                       99120 non-null float64
    mths_since_rcnt_il                96469 non-null float64
    total_bal_il                      99120 non-null float64
    il_util                           85480 non-null float64
    open_rv_12m                       99120 non-null float64
    open_rv_24m                       99120 non-null float64
    max_bal_bc                        99120 non-null float64
    all_util                          99114 non-null float64
    total_rev_hi_lim                  99120 non-null float64
    inq_fi                            99120 non-null float64
    total_cu_tl                       99120 non-null float64
    inq_last_12m                      99120 non-null float64
    acc_open_past_24mths              99120 non-null float64
    avg_cur_bal                       99120 non-null float64
    bc_open_to_buy                    98010 non-null float64
    bc_util                           97971 non-null float64
    chargeoff_within_12_mths          99120 non-null float64
    delinq_amnt                       99120 non-null float64
    mo_sin_old_il_acct                96469 non-null float64
    mo_sin_old_rev_tl_op              99120 non-null float64
    mo_sin_rcnt_rev_tl_op             99120 non-null float64
    mo_sin_rcnt_tl                    99120 non-null float64
    mort_acc                          99120 non-null float64
    mths_since_recent_bc              98067 non-null float64
    mths_since_recent_bc_dlq          26018 non-null float64
    mths_since_recent_inq             89254 non-null float64
    mths_since_recent_revol_delinq    36606 non-null float64
    num_accts_ever_120_pd             99120 non-null float64
    num_actv_bc_tl                    99120 non-null float64
    num_actv_rev_tl                   99120 non-null float64
    num_bc_sats                       99120 non-null float64
    num_bc_tl                         99120 non-null float64
    num_il_tl                         99120 non-null float64
    num_op_rev_tl                     99120 non-null float64
    num_rev_accts                     99120 non-null float64
    num_rev_tl_bal_gt_0               99120 non-null float64
    num_sats                          99120 non-null float64
    num_tl_120dpd_2m                  95661 non-null float64
    num_tl_30dpd                      99120 non-null float64
    num_tl_90g_dpd_24m                99120 non-null float64
    num_tl_op_past_12m                99120 non-null float64
    pct_tl_nvr_dlq                    99120 non-null float64
    percent_bc_gt_75                  98006 non-null float64
    pub_rec_bankruptcies              99120 non-null float64
    tax_liens                         99120 non-null float64
    tot_hi_cred_lim                   99120 non-null float64
    total_bal_ex_mort                 99120 non-null float64
    total_bc_limit                    99120 non-null float64
    total_il_high_credit_limit        99120 non-null float64
    dtypes: float64(74), int32(1), object(6)
    memory usage: 64.1+ MB
    None
'''

df.drop('annual_inc_joint', 1, inplace=True)
df.drop('dti_joint', 1, inplace=True)

print(df.select_dtypes(include=['int']).describe().T.assign(
    missing_pct=df.apply(lambda x: (len(x) - x.count()) / float(len(x)))))
'''
                  count      mean       std  min  25%  50%   75%   max  \
    emp_length  99120.0  5.757092  3.770359  0.0  2.0  6.0  10.0  10.0   

                missing_pct  
    emp_length          0.0  
'''

print(df['loan_status'].value_counts())
'''
    Current               79445
    Fully Paid            13066
    Charged Off            2502
    Late (31-120 days)     2245
    In Grace Period        1407
    Late (16-30 days)       454
    Default                   1
    Name: loan_status, dtype: int64
'''

# 标注优良用户二分类情况
df.loan_status.replace('Fully Paid', int(1), inplace=True)
df.loan_status.replace('Current', int(1), inplace=True)
df.loan_status.replace('Late (16-30 days)', int(0), inplace=True)
df.loan_status.replace('Late (31-120 days)', int(0), inplace=True)
df.loan_status.replace('Charged Off', np.nan, inplace=True)
df.loan_status.replace('In Grace Period', np.nan, inplace=True)
df.loan_status.replace('Default', np.nan, inplace=True)

print(df.loan_status.value_counts())
'''
    1.0    92511
    0.0     2699
    Name: loan_status, dtype: int64
'''

df.dropna(subset=['loan_status'], inplace=True)

# 相关度计算
cor = df.corr()
cor.loc[:, :] = np.tril(cor, k=-1)
cor = cor.stack()
print(cor[(cor > 0.55) | (cor < -0.55)])
'''
    funded_amnt                     loan_amnt                      1.000000
    funded_amnt_inv                 loan_amnt                      0.999994
                                    funded_amnt                    0.999994
    installment                     loan_amnt                      0.953380
                                    funded_amnt                    0.953380
                                    funded_amnt_inv                0.953293
    mths_since_last_delinq          delinq_2yrs                   -0.551275
    total_acc                       open_acc                       0.722950
    mths_since_last_major_derog     mths_since_last_delinq         0.685642
    open_il_24m                     open_il_12m                    0.760219
    total_bal_il                    open_il_6m                     0.566551
    open_rv_12m                     open_acc_6m                    0.623975
    open_rv_24m                     open_rv_12m                    0.774954
    max_bal_bc                      revol_bal                      0.551409
    all_util                        il_util                        0.594925
    total_rev_hi_lim                revol_bal                      0.815351
    inq_last_12m                    inq_fi                         0.563011
    acc_open_past_24mths            open_acc_6m                    0.553181
                                    open_il_24m                    0.570853
                                    open_rv_12m                    0.657606
                                    open_rv_24m                    0.848964
    avg_cur_bal                     tot_cur_bal                    0.828457
    bc_open_to_buy                  total_rev_hi_lim               0.626380
    bc_util                         all_util                       0.569469
    mo_sin_rcnt_tl                  mo_sin_rcnt_rev_tl_op          0.606065
    mort_acc                        tot_cur_bal                    0.551198
    mths_since_recent_bc            mo_sin_rcnt_rev_tl_op          0.614262
    mths_since_recent_bc_dlq        mths_since_last_delinq         0.751613
                                    mths_since_last_major_derog    0.553022
    mths_since_recent_revol_delinq  mths_since_last_delinq         0.853573
                                                                     ...   
    num_sats                        total_acc                      0.720022
                                    num_actv_bc_tl                 0.552957
                                    num_actv_rev_tl                0.665429
                                    num_bc_sats                    0.630778
                                    num_op_rev_tl                  0.826946
                                    num_rev_accts                  0.663595
                                    num_rev_tl_bal_gt_0            0.668573
    num_tl_30dpd                    acc_now_delinq                 0.801444
    num_tl_90g_dpd_24m              delinq_2yrs                    0.669267
    num_tl_op_past_12m              open_acc_6m                    0.722131
                                    open_il_12m                    0.557902
                                    open_rv_12m                    0.844841
                                    open_rv_24m                    0.660265
                                    acc_open_past_24mths           0.774867
    pct_tl_nvr_dlq                  num_accts_ever_120_pd         -0.592502
    percent_bc_gt_75                bc_util                        0.844108
    pub_rec_bankruptcies            pub_rec                        0.580798
    tax_liens                       pub_rec                        0.752084
    tot_hi_cred_lim                 tot_cur_bal                    0.982693
                                    avg_cur_bal                    0.795652
                                    mort_acc                       0.560840
    total_bal_ex_mort               total_bal_il                   0.902486
    total_bc_limit                  max_bal_bc                     0.581536
                                    total_rev_hi_lim               0.775151
                                    bc_open_to_buy                 0.834159
                                    num_bc_sats                    0.633461
    total_il_high_credit_limit      open_il_6m                     0.552023
                                    total_bal_il                   0.960349
                                    num_il_tl                      0.583329
                                    total_bal_ex_mort              0.889238
    Length: 93, dtype: float64
'''

# 模型构建阶段
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.preprocessing import OneHotEncoder

# https://ljalphabeta.gitbooks.io/python-/content/categorical_data.html

X = df.drop('loan_status', 1, inplace=False)
print(X)  # [95210 rows x 78 columns]

Y = df.loan_status
print(Y)  # [95210 rows x 1 columns]

# 填充空数据
X.fillna(0.0, inplace=True)
X.fillna(0, inplace=True)

# 训练集&测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=123)

# 查看数据集尺寸
print(x_train.shape)  # (66647, 78)
print(y_train.shape)  # (66647,)
print(x_test.shape)  # (28563, 78)
print(y_test.shape)  # (28563,)

print(y_train.value_counts())
'''
    1.0    64712
    0.0     1935
    Name: loan_status, dtype: int64
'''
print(y_test.value_counts())
'''
    1.0    27799
    0.0      764
    Name: loan_status, dtype: int64
'''

# GBRT模型
param_grid = {'learning_rate': [0.1],
              'max_depth': [2],
              'min_samples_split': [50, 100],
              'n_estimators': [100, 200]
              }

est = GridSearchCV(ensemble.GradientBoostingRegressor(),
                   param_grid, n_jobs=4, refit=True)

est.fit(x_train, y_train)

best_params = est.best_params_

print(best_params)  # {'min_samples_split': 100, 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}

# 评估模型
est = ensemble.GradientBoostingRegressor(min_samples_split=50, n_estimators=300,
                                         learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(x_train,
                                                                                                        y_train)
'''
    CPU times: user 24.2 s, sys: 251 ms, total: 24.4 s
    Wall time: 25.6 s
'''

print(est.score(x_test, y_test))  # 0.028311715416075908

est = ensemble.GradientBoostingRegressor(min_samples_split=50, n_estimators=100,
                                         learning_rate=0.1, max_depth=2, random_state=0, loss='ls').fit(x_train,
                                                                                                        y_train)
print(est.score(x_test, y_test))  # 0.029210266192750467


def compute_ks(data):
    sorted_list = data.sort_values(['predict'], ascending=[True])
    total_bad = sorted_list['label'].sum(axis=None, skipna=None, level=None, numeric_only=None) / 3
    total_good = sorted_list.shape[0] - total_bad
    max_ks = 0.0
    good_count = 0.0
    bad_count = 0.0
    for index, row in sorted_list.iterrows():
        if row['label'] == 3:
            bad_count += 1.0
        else:
            good_count += 1.0
        val = bad_count / total_bad - good_count / total_good
        max_ks = max(max_ks, val)
    return max_ks


test_pd = pd.DataFrame()
test_pd['predict'] = est.predict(x_test)
test_pd['label'] = y_test
print(compute_ks(test_pd[['label', 'predict']]))  # 0.0

# 前10个最重要的权重字段
feature_importance = est.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

indices = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10), feature_importance[indices], color='dodgerblue', alpha=.4)
plt.yticks(np.arange(10 + 0.25), np.array(X.columns)[indices])
_ = plt.xlabel('Relative importance'), plt.title('Top Ten Important Variables')

# XGBoost
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

clf2 = xgb.XGBClassifier(n_estimators=50, max_depth=1,
                         learning_rate=0.01, subsample=0.8, colsample_bytree=0.3, scale_pos_weight=3.0,
                         silent=True, nthread=-1, seed=0, missing=None, objective='binary:logistic',
                         reg_alpha=1, reg_lambda=1,
                         gamma=0, min_child_weight=1,
                         max_delta_step=0, base_score=0.5)

clf2.fit(x_train, y_train)
print(clf2.score(x_test, y_test))  # 0.973252109372

test_pd2 = pd.DataFrame()
test_pd2['predict'] = clf2.predict(x_test)
test_pd2['label'] = y_test
print(compute_ks(test_pd[['label', 'predict']]))  # 0.0
print(clf2.feature_importances_)
'''
    [ 0.          0.30769232  0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.05128205
      0.          0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.          0.
      0.05128205  0.30769232  0.2820513   0.          0.          0.          0.
      0.        ]
'''

# 数去前10个重要字段分布
feature_importance = clf2.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

indices = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10), feature_importance[indices], color='dodgerblue', alpha=.4)
plt.yticks(np.arange(10 + 0.25), np.array(X.columns)[indices])
_ = plt.xlabel('Relative importance'), plt.title('Top Ten Important Variables')

# 随机森林
clf3 = RandomForestRegressor(n_jobs=-1, max_depth=10, random_state=0)
clf3.fit(x_train, y_train)
print(clf3.score(x_test, y_test))  # 0.0148713087517

test_pd3 = pd.DataFrame()
test_pd3['predict'] = clf3.predict(x_test)
test_pd3['label'] = y_test
print(compute_ks(test_pd[['label', 'predict']]))  # 0.0
print(clf3.feature_importances_)

# Top Ten
feature_importance = clf3.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
'''
    [ 0.02588781  0.10778862  0.00734994  0.02090219  0.02231172  0.00778016
      0.00556834  0.01097013  0.00734689  0.0017027   0.00622544  0.01140843
      0.00530896  0.00031185  0.01135318  0.          0.01488991  0.01840559
      0.00585621  0.00652523  0.0066759   0.00727607  0.00955013  0.01004672
      0.01785864  0.00855197  0.00985739  0.01477432  0.02184904  0.01816184
      0.00878854  0.02078236  0.01310288  0.00844302  0.01596395  0.01825196
      0.01817367  0.00297759  0.00084823  0.02808718  0.02917066  0.00897034
      0.01139324  0.01532409  0.01467681  0.0032855   0.01066291  0.00581661
      0.00955357  0.00417743  0.01333577  0.00489264  0.0128039   0.01340195
      0.01286394  0.01619219  0.00395603  0.00508973  0.          0.00234757
      0.00378329  0.00502684  0.01732834  0.01178674  0.00030035  0.01189509
      0.00942532  0.00841645  0.01571355  0.00288054  0.          0.0011667
      0.00106548  0.00488734  0.          0.00200132  0.00062765  0.04130873
      0.10076558  0.00022293  0.00165858  0.00308408  0.0008255   0.        ]
'''

# XTR
clf4 = ExtraTreesRegressor(n_jobs=-1, max_depth=10, random_state=0)
clf4.fit(x_train, y_train)
print(clf4.score(x_test, y_test))  # 0.020808034579

test_pd4 = pd.DataFrame()
test_pd4['predict'] = clf4.predict(x_test)
test_pd4['label'] = y_test
print(compute_ks(test_pd[['label', 'predict']]))  # 0.0
print(clf4.feature_importances_)
'''
    [ 0.00950112  0.17496689  0.00476969  0.00538677  0.00898343  0.01604885
      0.0139889   0.00605683  0.0042762   0.00358536  0.0144985   0.00915189
      0.00643305  0.00637134  0.0050764   0.00218012  0.00925068  0.00363339
      0.00988441  0.00645297  0.00662444  0.00934969  0.00739012  0.00635592
      0.00633908  0.00923972  0.01263829  0.01190224  0.00914159  0.00402144
      0.00917841  0.01456563  0.01161155  0.01097394  0.00506868  0.00772159
      0.00560163  0.01132941  0.00172528  0.0085601   0.01282485  0.00970629
      0.00956066  0.00731205  0.02087289  0.00430205  0.0062769   0.00765693
      0.00922104  0.00296456  0.00563208  0.00459181  0.0133819   0.00548208
      0.00450864  0.0132415   0.00677772  0.00509891  0.00108962  0.00578448
      0.00934323  0.00715127  0.01078137  0.00855071  0.00695096  0.01488993
      0.00317962  0.00485367  0.00476553  0.00509674  0.          0.00733654
      0.00097223  0.00380448  0.00534715  0.00356893  0.0128526   0.11944538
      0.11758343  0.00195945  0.00225379  0.00243429  0.0007562   0.        ]
'''

# Top Ten
feature_importance = clf4.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

indices = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10), feature_importance[indices], color='dodgerblue', alpha=.4)
plt.yticks(np.arange(10 + 0.25), np.array(X.columns)[indices])
_ = plt.xlabel('Relative importance'), plt.title('Top Ten Important Variables')

indices = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10), feature_importance[indices], color='dodgerblue', alpha=.4)
plt.yticks(np.arange(10 + 0.25), np.array(X.columns)[indices])
_ = plt.xlabel('Relative importance'), plt.title('Top Ten Important Variables')
