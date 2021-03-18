import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import scipy.stats as st
import itertools
from math import exp, sqrt, log
from random import gauss, seed
from time import time

def Black_Scholes_formula(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    p = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
    return p

def Monte_Carlo_simulation1(S, r, T, sigma,N,delta_t,I):
    S_temp_1 = []
    for i in range(I):
        path1 = []
        for t in range(N+1):
            if t==0:
                path1.append(S)
            else:
                z = gauss(0.0,1.0)
                Delta_S = r * path1[t-1] * delta_t + sigma * path1[t-1] * z * np.sqrt(delta_t)
                # Stock maturity index level
                S_T = path1[t-1] + Delta_S
                path1.append(S_T)
        S_temp_1.append(path1)
    # Present value of options
    C0 = exp(-r * T) * sum([max(path1[-1] - K,0) for path1 in S_temp_1])/I
    return C0

def Monte_Carlo_simulation2(S, r, T, sigma, N,I):
    S_temp_2 = []
    for i in range(I):
        path2 = []
        for t in range(N+1):
            if t==0:
                path2.append(S)
            else:
                z = gauss(0.0,1.0)
                # Stock maturity index level
                S_T = path2[t-1] * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
                path2.append(S_T)
        S_temp_2.append(path2)
    # Present value of options
    C0 = exp(-r * T) * sum([max(path2[-1] - K,0) for path2 in S_temp_2])/I
    return C0


if __name__ == "__main__":
    stock_Ui = np.zeros([18, 127])
    Realized_volatility = np.zeros(18)
    path = r'C:\Users\hasee\Desktop\CUHK\CMSC5718-Computational Finance\assignment_2_stock_data.xlsx'
    sheet = pd.read_excel(path, sheet_name='stock prices')
    sheet2 = pd.read_excel(path, sheet_name='implied volatility')
    for i in range(3, 21):
        # get one stock data
        single_stock_data = sheet.iloc[5:133, i]
        # print(single_stock_data)
        for j in range(5, 132):
            stock_Ui[i - 3][j - 5] = math.log(single_stock_data[j + 1] / single_stock_data[j])

    for i in range(18):
        each_stock_data = stock_Ui[i, :]
        annualized_standard_deviation = np.std(each_stock_data, ddof=1) * np.sqrt(252)
        Realized_volatility[i] = annualized_standard_deviation

    print(
        '\n-------------------------------------------------------Part1-1-i--------------------------------------------------------------\n')
    print("Realized_volatility: ", Realized_volatility)

    # Stock X= China Construction Bank (939)
    temp = sheet.iloc[5:133, 8]
    S = temp[132]
    K = S
    T = 0.25
    r = 0.0075
    Sigma1 = Realized_volatility[5]
    list_temp = sheet2.iloc[9:10, 5]
    Sigma2 = list_temp[9]

    C1 = Black_Scholes_formula(S, K, T, r, Sigma1)
    C2 = Black_Scholes_formula(S, K, T, r, Sigma2)

    print(
        '\n-------------------------------------------------------Part1-1-ii--------------------------------------------------------------\n')
    print("European call option based on realized volatility: ", C1)
    print("European call option based on implied volatility: ", C2)

    N = 90
    t = T / N
    I1 = 1000
    I2 = 10000
    I3 = 100000
    I4 = 500000

    C1 = Monte_Carlo_simulation1(S, r, T, Sigma1, N,t,I1)
    C2 = Monte_Carlo_simulation1(S, r, T, Sigma2, N,t,I1)
    C3 = Monte_Carlo_simulation1(S, r, T, Sigma1, N,t,I2)
    C4 = Monte_Carlo_simulation1(S, r, T, Sigma2, N,t,I2)
    C5 = Monte_Carlo_simulation1(S, r, T, Sigma1, N,t,I3)
    C6 = Monte_Carlo_simulation1(S, r, T, Sigma2, N,t,I3)
    C7 = Monte_Carlo_simulation1(S, r, T, Sigma1, N,t,I4)
    C8 = Monte_Carlo_simulation1(S, r, T, Sigma2, N,t,I4)
    print(
        '\n-------------------------------------------------------Part1-2-i--------------------------------------------------------------\n')
    print("European Option price with 1000 paths based on realized volatility:", C1)
    print("European Option price with 1000 paths based on implied volatility:", C2)
    print("European Option price with 10000 paths based on realized volatility:", C3)
    print("European Option price with 10000 paths based on implied volatility:", C4)
    print("European Option price with 100000 paths based on realized volatility:", C5)
    print("European Option price with 100000 paths based on implied volatility:", C6)
    print("European Option price with 500000 paths based on realized volatility:", C7)
    print("European Option price with 500000 paths based on implied volatility:", C8)

    N1 = 1
    t1 = T / N1

    C9 = Monte_Carlo_simulation2(S, r, T, Sigma1, N1,I1)
    C10 = Monte_Carlo_simulation2(S, r, T, Sigma2, N1,I1)
    C11 = Monte_Carlo_simulation2(S, r, T, Sigma1, N1,I2)
    C12 = Monte_Carlo_simulation2(S, r, T, Sigma2, N1,I2)
    C13 = Monte_Carlo_simulation2(S, r, T, Sigma1, N1,I3)
    C14 = Monte_Carlo_simulation2(S, r, T, Sigma2, N1,I3)
    C15 = Monte_Carlo_simulation2(S, r, T, Sigma1, N1,I4)
    C16 = Monte_Carlo_simulation2(S, r, T, Sigma2, N1,I4)

    print(
        '\n-------------------------------------------------------Part1-2-ii--------------------------------------------------------------\n')
    print("European Option price with 1000 paths based on realized volatility:", C9)
    print("European Option price with 1000 paths based on implied volatility:", C10)
    print("European Option price with 10000 paths based on realized volatility:", C11)
    print("European Option price with 10000 paths based on implied volatility:", C12)
    print("European Option price with 100000 paths based on realized volatility:", C13)
    print("European Option price with 100000 paths based on implied volatility:", C14)
    print("European Option price with 500000 paths based on realized volatility:", C15)
    print("European Option price with 500000 paths based on implied volatility:", C16)
