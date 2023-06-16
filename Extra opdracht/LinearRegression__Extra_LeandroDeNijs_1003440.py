'''
Program to scrape data from a website and use it to predict life expectancy.
Extra assignment from https://wiztech.nl/hr/ti/tinlab_ml/opdrachten/extra_opdracht.pdf
Author: Leandro de Nijs

Goals:
1. Verzamelen van gegevens vanaf een dynamisch gegenereerde webpagina. 
-Done

2. Prepareren (converteren, filteren (“schoonmaken”), completeren, uniformeren) van die
gegevens. 
-Done, filtered outliers, calculated BMI from mass and length and removed mass and length columns.

3. Transformeren van die gegevens naar een vorm die zich leent voor lineaire regressie, waarbij de
levensverwachting in jaren wordt voorspeld uit aanleg en levensstijl. 
-Done, calculated BMI and split the table into X (variables) and Y (life expectancy)

4. Uitvoeren van zo’n regressie, eerst met “bare bones” NumPy, daarna met een dedicated
regressor uit SciKitLearn.
-Done, Implemented 3 regressors. used normal equation from NumPy, LinearRegression from SciKitLearn and MLPRegressor from SciKitLearn

5. Grafisch representeren van de resultaten met behulp van de Python library MatPlotLib. Hierbij
kunnen vooral bar charts en pie charts van pas komen. Line charts liggen minder voor de hand
maar kunnen een nuttige aanvulling zijn.
-Done, used bar chart to visualize the weights of the variables.
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By

from sklearn import linear_model
from sklearn.neural_network import MLPRegressor

from scipy import stats


class Scraper:
    '''Class to scrape data from a website'''
    def __init__(self, url):
        '''
        Initialize the scraper
        :param url: url of the website to scrape
        '''

        self.data = []
        self.driver = webdriver.Firefox()
        self.driver.get(url)
    
    def scrapeTable(self):
        '''
        Scrape the table from the website
        :return: pandas dataframe with the data from the table
        '''

        # Retrieve the table element from the website
        table = self.driver.find_element(By.XPATH, "//table")
        rows = table.find_elements(By.TAG_NAME, 'tr')
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, 'td')
            rowdata = []
            for col in cols:
                rowdata.append(col.text)
            self.data.append(rowdata)

        # Convert the data to a pandas dataframe
        df = pd.DataFrame(self.data[1:], columns=self.data[0]).astype(float)
        return df
    
    def close(self):
        '''Close the webdriver'''
        self.driver.close()

class LinearRegression:
    '''Class to perform linear regression'''

    def normalEquation(self, X, Y):
        '''Normal equation to calculate the weights of the variables
        :param X: variables
        :param Y: life expectancy
        :return: weights of the variables
        '''
        beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
        return beta
    
    def scikitLearn(self, X, Y):
        '''Linear regression using scikit learn
        :param X: variables
        :param Y: life expectancy
        :return: weights of the variables
        '''
        reg = linear_model.LinearRegression()
        reg.fit(X, Y)
        beta = reg.coef_.reshape(-1,1)
        return beta
    
    def scikitMLP(self, X, Y):
        '''Linear regression using scikit MLPregressor
        :param X: variables
        :param Y: life expectancy
        :return: regressor
        '''
        reg = MLPRegressor(hidden_layer_sizes=(10, 10, 10), max_iter=500, activation='relu', solver='adam', random_state=1)
        reg.fit(X.values, Y.values.flatten())
        return reg

class DataProcessing:
    '''Class to process the data'''

    def cleanData(self, df):
        '''Remove outliers from the data using z-score
        :param df: dataframe with the data
        :return: dataframe without outliers
        '''

        df = df[(np.abs(stats.zscore(df, axis=0))<3).all(axis=1)]
        df.reset_index(inplace=True)
        return df
    
    def calculateBMI(self, df):
        '''Calculate the BMI from the mass and length and remove the mass and length columns
        :param df: dataframe with the data
        '''

        BMI = df['mass'] / (df['length']**2)
        df['BMI'] = BMI 
        del df['mass']
        del df['length']

    def splitTable(self, df):
        '''Split the table into X (variables) and Y (life expectancy)
        :param df: dataframe with the data
        :return: X and Y
        '''

        X = df.iloc[:, :-1]
        Y = df.iloc[:, -1:]
        return X, Y
    
def visualize(X, Y, pltNum):
    '''Visualize the weights of the variables
    :param X: variables
    :param Y: life expecDetancy
    :param pltNum: number of the plot
    '''

    plt.figure(pltNum, figsize=(10,5))
    variables = Y
    print(variables)
    weights = X.reshape(-1)
    plt.bar(variables,weights)

if __name__ == "__main__":
    # Try to load the data from a pickle file, if it doesn't exist, scrape the data from the website
    try:
        data = pd.read_pickle("data.pkl")
        data = DataProcessing().cleanData(data)
        print(data)
    except:
        print("No data found, scraping data from website")
        scraper = Scraper("https://wiztech.nl/mitw/mcr/data_generator.html")
        data = scraper.scrapeTable()
        print(data.shape[0])
        data = DataProcessing().cleanData(data)
        print(data.shape[0])
        data.to_pickle("data.pkl")
        scraper.close()

    X, Y = DataProcessing().splitTable(data)
    print(X.head())
    print(Y.head())

    DataProcessing().calculateBMI(X)

    beta = LinearRegression().normalEquation(X,Y)
    print("Beta")
    print(beta)

    beta2 = LinearRegression().scikitLearn(X,Y)
    print("Beta2")
    print(beta2)

    beta3 = LinearRegression().scikitMLP(X,Y)
    print("Beta3")
    print(beta3.coefs_)

    # Testing the 3 models
    # Test the model with the first row of the data using numpy
    test = X.loc[0]
    test = np.dot(test, beta)

    # Test the model with the first row of the data using scikit learn
    test2 = X.loc[0]
    test2 = np.dot(test2, beta2)

    # Test the model with the first row of the data using scikit MLP
    test3 = X.loc[0]
    test3 = beta3.predict(test3.values.flatten().reshape(1, -1))

    print()
    print("Calculated life expectancy Numpy:")
    print(test)
    print("Calculated life expectancy Scikit-learn:")
    print(test2)
    print("Calculated life expectancy Scikit-MLP:")
    print(test3)
    print("Actual life expectancy:")
    print(Y.loc[0])

    visualize(beta.reshape(-1)[1:], X.columns[1:], 1)
    visualize(beta2.reshape(-1)[1:], X.columns[1:], 2)
    plt.show()