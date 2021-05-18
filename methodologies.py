# -*- coding: utf-8 -*-
# @file methodologies.py
# @author Eilin Luna 
#**************************************************************************

import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from scipy.stats import norm, kurtosis, t, chi2

class methodology(object):
    
    """Compute the different methodologies for the VaR of a portfolio

    Parameters
    -----------
    stocks : list
        The list of assets symbols
        ["stock1", "stock2", "stock3", ...]
        
    amount : list
        The list of amounts to invest in each asset.
        [100, 200, 300, ...]
        
    init : str
        Historical data start date.
        "01/01/2020"
        
    end : str
        Historical data end date.
        "17/05/2021"
        
    alphas : list
        Levels of significance.
        [0.1, 0.025, 0.05, ...]
        
    periods : int
        Number of days for which the VaR will be calculated.
        Default: 277

    Returns
    -------
    DataFrame (names=(Fechas, p=0.1, p=0.05, p=0.025, 0.01))
        A dataframe containing VaR according to the selected method.
    """
    
    def __init__(self, stocks, amount, init, end, alphas, periods=277):
        # Instances
        self.stocks = stocks   
        self.amount = amount
        self.start = init
        self.end = end
        self.alphas = alphas   
        self.periods = periods  
        # Amount of assets
        self.n = len(stocks) 
        # Weights vector
        self.W = np.array(amount)/np.sum(amount)  
        # Data : Date, PriceStock1, PriceStock2, ..., PriceStockn
        self.data = self.get_data()  
        # Ret : Date, ReturnStock1, ReturnStock2,...,ReturnStockn  
        self.Rs = self.get_return() 
        # Retp: Date, ReturnPortfolio
        self.Rp = self.get_performance() 
        # Amount of data
        self.N = len(self.data)-1  
        # Business days and previous to use moving windows.
        self.days = self.N - self.periods 
        # Dates on which the VaR will be calculated
        self.dates = list(self.get_dates()) + ["VaR t+1"]
        # Number where init first moving window.
        self.a = self.N-self.periods-self.days 
        # Number where end first moving window.
        self.b = self.N-self.periods  
        # To calculate standard desviation
        self.landa = 0.94 
        
    # Downloading values of each asset in Yahoo Finance
    def get_data(self):
        if self.stocks!=[]:
            try:
                data = pdr.get_data_yahoo(self.stocks, 
                                          start = self.start, 
                                          end = self.end)
                return data["Close"]
            except:
                print("Error")
                return
    
    # Get the returns: R=log(Pf/Pi)
    def get_return(self):
        df = pd.DataFrame()
        for i in self.stocks:
            Pf = self.data[i][1:].values
            Pi = self.data[i][:-1].values
            df[i] = np.log(Pf/Pi)
        df.index = [k.strftime("%d/%m/%Y") for k in self.data.index[1:]]
        return df
    
    # Some statistics for the performance of each asset
    def get_statistics(self):
        df = pd.DataFrame()
        df["Participacion"] = self.W 
        df.index = self.stocks
        df["Precio Cierre"] = self.data.iloc[-1].values
        df["Rendimiento"]=self.Rs.iloc[-1]
        df["Media"]=self.Rs.mean().values
        df["Desv. estándar"]=self.Rs.std().values
        df["Varianza"]=self.Rs.var().values
        df["Curtosis"]=self.Rs.apply(kurtosis).values
        df["Mínimo"]=self.Rs.min().values
        df["Máximo"]=self.Rs.max().values
        df["Cantidad"]=self.Rs.count().values
        df.sort_values(by="Participacion", ascending=False, inplace=True)
        for column in ["Participacion","Rendimiento","Media","Desv. estándar",
                       "Varianza", "Mínimo", "Máximo"]:
            df[column] = (df[column]*100).round(3).astype(str) + '%'
        return df.transpose()
    
    # To return the Dataframes.
    def apply_style(self, df):
        # Place the VaR t+1 in the first position
        df=df.reindex([df.index[-1]] + list(df.index[:-1])).reset_index(drop=True)

        # Convert the data to percentages
        for i in df.columns[1:]: 
            df[i]=(df[i]*100).round(4).astype(str)+"%" 
        
        return df
        
    # Computing return of the portfolio
    def get_performance(self):
        R = (self.Rs*self.W).sum(axis=1)
        return pd.DataFrame(R, columns=["Returns"])
    
    # Extracting dates in the chosen period (277 days (by default))
    def get_dates(self):
        Dates = self.data[self.N-self.periods+1:self.N+1].index
        Dates = [i.strftime("%d/%m/%Y") for i in Dates]
        return Dates
    
    # Calculating standard deviation by moving windows
    def get_sd_1(self, s):
        return np.array([np.std(self.Rs[s][self.a+j:self.b+j], ddof=1) for j in range(self.periods+1)])
    
    # Calculating standard deviation as: d.j+1 = sqrt{lamda*(d.j**2) + (1-lamda)*(Ri**2)}
    def get_sd_2(self, s):
        r0 = self.Rs[s][self.N-self.periods:self.N]
        d0 = [np.std(self.Rs[s][self.a:self.b], ddof=1)]
        for j in range(self.periods):
            d0 += [np.sqrt(self.landa*d0[j]**2+(1-self.landa)*r0[j]**2)]
        return np.array(d0)

    #  Calculating standard deviation as: sqrt{v-2/v}*sd1
    def get_sd_3(self, s):
        v = self.get_degrees_of_freedom(self.Rs[s])
        d = self.get_sd_1(s)
        return np.sqrt((v-2)/v)*d

    # Calculating standard deviation as: d.j+1 = sqrt{v.j-2/v.j}*sqrt{lamda*(d.j**2) + (1-lamda)*(Ri**2)}
    def get_sd_4(self, s):
        v = self.get_degrees_of_freedom(self.Rs[s])
        r0 = self.Rs[s][self.N-self.periods:self.N]
        d0 = [np.std(self.Rs[s][self.a:self.b]*np.sqrt((v[0]-2)/v[0]), ddof=1)]
        for j in range(self.periods):
            d0 += [np.sqrt(self.landa*d0[j]**2+(1-self.landa)*r0[j]**2)*np.sqrt((v[j+1]-2)/v[j+1])]
        return np.array(d0)
    
    # Calculating standard deviation as: sqrt{W*cov*W.T} mobile
    def get_sd_5(self):
        W = np.matrix(self.W)
        desv = []
        for j in range(self.periods+1):
            rent_movil = np.transpose(self.Rs[:][self.a+j:self.b+j].values)
            cov = np.matrix(np.cov(rent_movil))
            desv_p = W*cov*W.T
            desv += [np.sqrt(desv_p[0,0])]
        return np.array(desv)

    # Calculating moving arithmetic mean of an asset
    def get_mean(self, s):
        return [np.mean(self.Rs[s][self.a+j:self.b+j]) for j in range(self.periods)]

    # Calculating Degrees of Freedom given the returns
    def get_degrees_of_freedom(self, r):
        DF = []
        for j in range(self.periods+1):
            rango = r[self.a+j:self.b+j]
            k = kurtosis(rango)
            DF += [np.round((4*k+6)/k)]
        return np.array(DF, dtype=int)

    # VaR Delta Normal (s=Activo)   
    def VaRDeltaNormal(self,s,p=False):
        df = pd.DataFrame()
        df["Fechas"] = self.dates
        sd = self.get_sd_1(s)
        for a in self.alphas:
            Zalpha = -norm.ppf(1-a)
            df['p='+str(a)]=sd*Zalpha
        if p: return df
        return self.apply_style(df)

    # VaR Simulacion Univariada (s=Activo)   
    def VaRSimulacionHistoricaUnivariada(self,s):
        df = pd.DataFrame()
        df["Fechas"] = self.dates
        for a in self.alphas:
            df['p='+str(a)]=[np.percentile(self.Rs[s][self.a+j:self.b+j],a*100) for j in range(self.periods+1)]
        return self.apply_style(df)

    # VaR Delta T (s=Activo)
    def VaRDeltaT(self,s):
        df = pd.DataFrame()
        df["Fechas"] = self.dates
        v = self.get_degrees_of_freedom(self.Rs[s])
        d = self.get_sd_1(s)
        sd = np.sqrt((v-2)/v)*d
        for a in self.alphas:
            df['p='+str(a)]=sd*-t.ppf(1-a, v)
        return self.apply_style(df)
        
    # VaR Suma Simple (Portfolio)
    def VaRSumaSimple(self):
        df = pd.DataFrame()
        df["Fechas"] = self.dates
        for a in self.alphas:
            df['p='+str(a)]=sum([self.VaRDeltaNormal(i,p=True)['p='+str(a)] for i in self.stocks])
        return self.apply_style(df)

    # VaR Baricentro EWMA Normal (Portafolio)
    def VaRBaricentroEWMANormal(self):
        df = pd.DataFrame()
        df["Fechas"] = self.dates
        Desv_Bar_Prom_EWMA = sum([self.get_sd_2(self.stocks[i])*self.W[i] for i in range(self.n)])
        for a in self.alphas:
            df['p='+str(a)]=Desv_Bar_Prom_EWMA*-norm.ppf(1-a)
        return self.apply_style(df)

    # VaR Baricentro Promedio Normal (Portfolio)
    def VaRBaricentroPromedioNormal(self):
        df = pd.DataFrame()
        df["Fechas"] = self.dates
        Desv_Bar_Prom = sum([self.get_sd_1(self.stocks[i])*self.W[i] for i in range(self.n)])
        for a in self.alphas:
            df['p='+str(a)]=Desv_Bar_Prom*-norm.ppf(1-a)
        return self.apply_style(df)

    # VaR Simulacion Historica (Portfolio)
    def VaRSimulacionHistorica(self):
        df = pd.DataFrame()
        df["Fechas"] = self.dates
        for a in self.alphas:
            df['p='+str(a)]=[np.percentile(self.Rp[self.a+j:self.b+j],a*100) for j in range(self.periods+1)]
        return self.apply_style(df)   

    # VaR Baricentro Promedio T (Portafolio)
    def VaRBaricentroPromedioT(self):
        df = pd.DataFrame()
        df["Fechas"] = self.dates
        GL=self.get_degrees_of_freedom(self.Rp['Returns'])
        desv = sum([self.get_sd_3(self.stocks[i])*self.W[i] for i in range(self.n)])
        for a in self.alphas:
            df['p='+str(a)]=desv*-t.ppf(1-a, GL)
        return self.apply_style(df)

    # VaR Baricentro EWMA T (Portfolio)
    def VaRBaricentroEWMAT(self):
        df = pd.DataFrame()
        df["Fechas"] = self.dates
        GL=self.get_degrees_of_freedom(self.Rp['Returns'])
        desv = sum([self.get_sd_4(self.stocks[i])*self.W[i] for i in range(self.n)])
        for a in self.alphas:
            df['p='+str(a)]=desv*-t.ppf(1-a, GL)
        return self.apply_style(df)

    # VaR Matrix CovVar (Portfolio)
    def VaRMatrizVarCovar(self):
        df = pd.DataFrame()     
        df["Fechas"] = self.dates
        desv = self.get_sd_5()
        for a in self.alphas:
            Zalpha = -norm.ppf(1-a)
            df['p='+str(a)]=desv*Zalpha
        return self.apply_style(df)
    
    # M, dataframe returned by the method used.
    def exceptions(self, M):
        df = pd.DataFrame()
        df["Fechas"] = np.array(self.dates)[:-1]
        for a in self.alphas: M['p='+str(a)] = M['p='+str(a)].str.rstrip('%').astype('float')/100.
        RP = self.Rp['Returns'][self.N-self.periods:self.N].values
        for a in self.alphas:
            df['p='+str(a)]=RP<M['p='+str(a)][1:].values
        return df  
    
    def BackTesting(self, M):
        ex = self.exceptions(M)
        p = np.array(self.alphas)
        x = ex.sum()[1:].values
        m = ex.count()[1:].values
        p_est = x/m
        level = np.array([0.05]*len(self.alphas))
        BackTesting = m*p
        num = (p**x)*(1-p)**(m-x)
        den = (p_est**x)*(1-p_est)**(m-x)
        test_kupiec = -2*np.log(list(num/den))
        Pvalue = chi2.pdf(test_kupiec,1)
        zone = np.where(Pvalue<=level, "Rechazo H0", "No Rechazo H0")
        #**********************************************************************
        df = pd.DataFrame({'x':x , 
                          'm':m,
                          'p estimado':p_est , 
                          'nivel de significancia':level,
                          'test de kupiec':test_kupiec, 
                          'Valor P':Pvalue,
                          "Zona de rechazo":zone, 
                          'BackTesting':BackTesting,
                          "Valor de eficiencia": 1-p_est},
                          index=["p="+str(pp) for pp in p])
        return df.T