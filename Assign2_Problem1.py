import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import scipy.stats as st
import itertools

def Black_Scholes_formula(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2=d1-(sigma*np.sqrt(T))
    p = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
    return p

stock_Ui = np.zeros([18, 127])
Realized_volatility = np.zeros(18)
path = r'C:\Users\hasee\Desktop\CUHK\CMSC5718-Computational Finance\assignment_2_stock_data.xlsx'
sheet = pd.read_excel(path, sheet_name='stock prices')
sheet2 = pd.read_excel(path, sheet_name='implied volatility')
for i in range(3, 21):
    # get one stock data
    single_stock_data = sheet.iloc[5:133, i]
    #print(single_stock_data)
    for j in range(5, 132):
        stock_Ui[i - 3][j - 5] = math.log(single_stock_data[j + 1] / single_stock_data[j])

for i in range(18):
    each_stock_data = stock_Ui[i, :]
    annualized_standard_deviation = np.std(each_stock_data, ddof=1) * np.sqrt(252)
    Realized_volatility[i] = annualized_standard_deviation

print(
    '\n-------------------------------------------------------Part1-1-i--------------------------------------------------------------\n')
print("Realized_volatility: ", Realized_volatility)


#Stock X= China Construction Bank (939)
temp= sheet.iloc[5:133, 8]
S=temp[132]
K=S
T=0.25
r=0.0075
Sigma1 = Realized_volatility[5]
list_temp = sheet2.iloc[9:10,5]
Sigma2 = list_temp[9]

C1 = Black_Scholes_formula(S, K, T, r, Sigma1)
C2 = Black_Scholes_formula(S, K, T, r, Sigma2)

print(
    '\n-------------------------------------------------------Part1-1-ii--------------------------------------------------------------\n')
print("European call option based on realized volatility: ", C1)
print("European call option based on implied volatility: ", C2)





