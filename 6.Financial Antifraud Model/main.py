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

# 保留前7列有用数据
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
# 向前补列
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

print(df.loan_amnt != df.funded_amnt).value_counts()

print(df.query('loan_amnt != funded_amnt').head(5))
