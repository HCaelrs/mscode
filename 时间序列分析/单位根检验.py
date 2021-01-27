from statsmodels.tsa.stattools import adfuller
import pandas as pd
"""
ADF��float
    ����ͳ�ơ�
pvalue��float
    probability value��MacKinnon����MacKinnon�Ľ���pֵ��1994�꣬2010�꣩��
usedlag��int
    ʹ�õ��ͺ�������
NOBS��int
    ����ADF�ع�ͼ����ٽ�ֵ�Ĺ۲�����
critical values��dict
    ����ͳ�����ݵ��ٽ�ֵΪ1����5����10��������MacKinnon��2010����
icbest��float
    ���autolag����None���������Ϣ��׼��
"""

df = pd.read_csv("ads.csv")

print(adfuller(df.Ads))

