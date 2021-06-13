############################################################################
#
# HydroPrecip.py
#
# machine learning trials to match urban lake hydrograph to precip history
# by Walt McNab, June 2021
#
############################################################################

from numpy import *
import pandas as pd
from scipy import interpolate
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


class Regressions:
    
    def __init__(self, dataSet, regList, pptSelect):
        # objects and parameters used for both training and validation data sets
        self.m, self.b, r, p, stderr = stats.linregress(dataSet['prior_1'].values, array(dataSet['hydrograph']))    # linear model in one variable
        self.reg = LinearRegression(fit_intercept=True)                                                 # multivariate linear model     
        self.rf = RandomForestRegressor(random_state=0, max_features='sqrt', criterion='mae', max_depth=6)           # random forest regression  
        self.svr = SVR(kernel='rbf', gamma=0.002, C=0.1)                                                # support vector machine regression
        self.ann = MLPRegressor(random_state=1, max_iter=500)
        self.regList = regList
        self.pptSelect = pptSelect
        
    def Train(self, dataSet, pptDefRecord):
        # populate training data set with regression results
        x = array(dataSet['prior_1'])   # linear model in one variable
        dataSet['linear'] = x*self.m + self.b                                                   
        self.reg.fit(pptDefRecord, array(dataSet['hydrograph']))          # multivariate linear model
        dataSet['MVLM'] = self.reg.predict(pptDefRecord)
        self.rf.fit(pptDefRecord, array(dataSet['hydrograph']))          # random forest regression
        dataSet['random forest'] = self.rf.predict(pptDefRecord)
        self.svr.fit(pptDefRecord, array(dataSet['hydrograph']))          # support vector machine regression
        dataSet['SVR'] = self.svr.predict(pptDefRecord)
        self.ann.fit(pptDefRecord, array(dataSet['hydrograph']))
        dataSet['ANN'] = self.ann.predict(pptDefRecord)
        self.PlotHydrographFits(dataSet, 'Training Set', 0)           # plot
        dataSet = self.Residuals(dataSet)                       # add residuals and return dataframe
        return dataSet
        
    def Validate(self, dataSet, pptDefRecord):
        # populate validation data set with predictions
        x = array(dataSet['prior_1'])
        dataSet['linear'] = x*self.m + self.b
        dataSet['MVLM'] = self.reg.predict(pptDefRecord)
        dataSet['random forest'] = self.rf.predict(pptDefRecord)
        dataSet['SVR'] = self.svr.predict(pptDefRecord)
        dataSet['ANN'] = self.ann.predict(pptDefRecord)
        self.PlotHydrographFits(dataSet, 'Validation Set', 1)           # plot
        dataSet = self.Residuals(dataSet)                       # add residuals and return dataframe        
        return dataSet

    def PlotHydrographFits(self, dataSet, title, iSet):
        # plot time series of model fits to hydrograph data
        setName = ['Training Set', 'Validation Set']
        for i, graph in enumerate(self.regList):
            title = setName[iSet] + ', Annual Precip. = ' + str(self.pptSelect*12)
            plt.figure(iSet*len(self.regList) + i)
            dataSet.plot(x='date', y=['hydrograph', graph], color=['blue', 'green'], title=title, ylim=(-2., 14.))

    def Residuals(self, dataSet):
        # compute residuals of model fits
        for model in self.regList:
            dataSet[model+' rsd'] = dataSet[model] - dataSet['hydrograph']
        return dataSet


class Params:
    
    def __init__(self):
        
        # read misc. params from file
        lineInput = []
        inputFile = open('params.txt','r')
        for line in inputFile:
            lineSplit = line.split()
            if len(lineSplit) <= 2: lineInput.append(line.split()[1])
            else: lineInput.append(line.split()[1:])
        inputFile.close()
        self.rainRecEndYear = int(lineInput[0]) 	# end of last full rainfall year in record
        self.corrStartDate = pd.to_datetime(lineInput[1]) 	# date window for mean rainfall correlation analysis
        self.corrEndDate = pd.to_datetime(lineInput[2])
        self.startTrainDate = pd.to_datetime(lineInput[3])    # beginning of training window  
        self.splitDate = pd.to_datetime(lineInput[4])	# date dividing training and validation sets
        self.lookBack = lineInput[5]     # array (of months for backward-looking rainfall deficit)
        self.pptSelect = float(lineInput[6])    # posited mean rainfall rate - monthly
        print('Read parameters.')    
    

class Rainfall:
    
    def __init__(self, params):
        
        # read precipitation data and develop metrics
        self.precipRecord = ReadData('precip_record.csv', 'precipitation')
        
        # annualized precipitation summary statistics        
        mu, sigma, numq = self.PrecipHistogram(params.rainRecEndYear)
        z = 1.96        # compute 95% confidence interval for mean value
        muRange = z*sigma/sqrt(numq)
        print('Processed precipitation data.')
        print('\tAnnual mean = ', mu)
        print('\tStandard deviation = ', sigma)
        print('\t95% confidence for mean, lower = ', mu-muRange)
        print('\t95% confidence for mean, upper = ', mu+muRange)

    def PrecipHistogram(self, endYear):
        # compute and plot cumulative distribution functions for annual precipitation
        annual = self.precipRecord[self.precipRecord['year']<=endYear][['year', 'precipitation']]
        annual = annual.groupby(['year']).sum()
        annual['percentile'] = annual.rank(pct=True)
        q = array(annual['precipitation'])
        alpha, locG, scaleG = stats.gamma.fit(q)        # fit CDFs
        locN, scaleN = stats.norm.fit(q)
        qRange = linspace(annual['precipitation'].min(), annual['precipitation'].max(), 50, endpoint=True)  # create CDF curves and plot
        fitPercentileG = stats.gamma.cdf(qRange, alpha, locG, scaleG)
        fitPercentileN = stats.norm.cdf(qRange, locN, scaleN)
        plt.figure(0) 
        plt.scatter(q, annual['percentile'], s=10, facecolors='none', edgecolors='black', label = 'data')
        plt.plot(qRange, fitPercentileG, color = 'blue', label = 'gamma fit')
        plt.plot(qRange, fitPercentileN, color = 'green', label = 'normal fit')
        plt.title('Precipitation CDF')
        plt.xlabel('Annual Rainfall (in/yr)')
        plt.ylabel('Cumulative Distribution')
        plt.legend(loc=4)
        plt.show()
        return mean(q), std(q), len(q)

    def FitHydrograph(self, hydroData, params):
        # estimate cumulative monthly rainfall deficit that maximizes correlation with hydrograph
        res = optimize.minimize_scalar(self.CorrelateDeficit, method='bounded', bounds = (0., 5.), args=(hydroData, params))
        pptOpt = res.x
        corrOpt = sqrt(1. - self.CorrelateDeficit(pptOpt, hydroData, params))
        print('Optimal fit of rainfall deficit to hydrograph:')
        print('\tAnnual rainfall, optimal = ', pptOpt*12)
        print('\tOptimal correlation coefficient = ', corrOpt)
        
    def CorrelateDeficit(self, pptMonthly, hydroData, params):
        # hydrograph versus rainfall deficit correlation coefficient
        self.precipRecord['deficit'] = cumsum(self.precipRecord['precipitation'] - pptMonthly)
        precipSubset = array(self.precipRecord[(self.precipRecord['date']>=params.corrStartDate)
            & (self.precipRecord['date']<=params.corrEndDate)]['deficit'])
        hydroSubset = array(hydroData[(hydroData['date']>=params.corrStartDate) 
            & (hydroData['date']<=params.corrEndDate)]['hydrograph'])
        R, p = stats.pearsonr(precipSubset, hydroSubset)
        return 1.0 - R**2        
    
    def LabelPrecip(self, numPrior):
        # month/year numbering index to permit backward-counting, timea-averaged rainfall deficit
        numPrecip = len(self.precipRecord)
        startIndex = -numPrior
        indexSeq = arange(startIndex, numPrecip+startIndex, 1)
        self.precipRecord['indexSeq'] = indexSeq
        self.precipRecord.set_index('indexSeq', inplace=True)

    def DistributeDeficit(self, hydroData, params):
        # distribute time-averaged precipitation deficits
        v = hydroData.index.values
        X = []          # labels for columns that will be used for machine learning routines
        for n in params.lookBack:
            cumRainDef = zeros(len(hydroData), float)
            for i in range(int(n)):
                cumRainDef = cumRainDef + array(self.precipRecord.loc[v-i]['deficit'])
            hydroData['prior_' + n] = cumRainDef / int(n)
            X.append('prior_' + n)
        return hydroData, X


### support functions ###


def ReadData(fileName, kind):
    # read time-dependent .csv file; merge different data sets by averaging by month
    dataSet = pd.read_csv(fileName)
    dataSet['date'] = pd.to_datetime(dataSet['date'])
    dataSet['month'] = pd.DatetimeIndex(dataSet['date']).month
    dataSet['year'] = pd.DatetimeIndex(dataSet['date']).year    
    dataSet = dataSet.groupby([kind, 'date'], as_index=False)[['month', 'year']].mean()
    dataSet['t'] = dataSet['date'].apply(lambda x: x.toordinal())
    dataSet.sort_values('date', inplace=True)
    dataSet.reset_index(inplace=True)
    dataSet.drop(['index'], axis=1, inplace=True)
    print('Read', kind, 'data.')
    return dataSet
    

def AssignHydro(hydroRaw, precipData):
    # interpolate hydrographs by month & year; use monthly precip record as a template
    x = array(hydroRaw['t'])
    y = array(hydroRaw['hydrograph'])
    f = interpolate.interp1d(x, y)
    startDay = hydroRaw['t'].min()
    endDay = hydroRaw['t'].max()
    numPrior = len(precipData[precipData['t']<startDay])  # months of precip data available prior to hydrograph
    precipSnippet = precipData[(precipData['t']>=startDay) & (precipData['t']<=endDay)]
    times = array(precipSnippet['t'])
    hydroInterp = f(times)
    hydroData = precipSnippet.copy()
    hydroData['hydrograph'] = hydroInterp
    hydroData.drop(['precipitation'], axis=1, inplace=True)
    hydroData.reset_index(drop=True, inplace=True)
    print('Interpolated hydrograph.')
    return hydroData, numPrior


def PlotResHistograms(trainData, validateData, regList, pptSelect):
    # compare distributions of fit residuals between training and validation sets
    bins = 20
    color = ['blue', 'green']
    label = ['Training', 'Validate']
    for j, model in enumerate(regList):
        plot_set = []
        plot_set.append(array(trainData[model+' rsd']))
        plot_set.append(array(validateData[model+' rsd']))    
        plt.figure(len(regList)*3 + j)
        for i, p in enumerate(plot_set):
            plt.hist(p, bins, color=color[i], label=label[i], edgecolor='black')
        plt.legend(loc='upper right')
        plt.xlim(-8., 8.)
        plt.xlabel('Residuals')
        plt.ylabel('N')
        plt.title(model + ', Annual Precip. = ' + str(pptSelect*12))
        plt.show()


def PlotScatter(trainData, validateData, regList, pptSelect):
    # compare modeled to observed histogram data in training and validation sets
    color = ['blue', 'green']
    label = ['Training', 'Validate']
    for j, model in enumerate(regList):
        x = []
        x.append(array(trainData['hydrograph']))
        x.append(array(validateData['hydrograph']))        
        y = []
        y.append(array(trainData[model]))
        y.append(array(validateData[model]))    
        plt.figure(len(regList)*4 + j)
        for i, p in enumerate(x):
            plt.scatter(x[i], y[i], s=10, facecolors=color[i], edgecolors=color[i], label=label[i])
        plt.xlim([-3., 14.])
        plt.ylim([-3., 14.])        
        plt.legend(loc='upper right')
        plt.xlabel('Observed Lake Stage')
        plt.ylabel('Modeled Lake Stage')
        plt.title(model + ', Annual Precip. = ' + str(pptSelect*12))
        plt.show()
    
    

def HydroPrecip():          ### main script ###

    params = Params()       # miscellaneous parameters
    
    rainfall = Rainfall(params)     # process precipitation data

    hydroRaw = ReadData('hydrograph.csv', 'hydrograph')                 # read and process hydrograph data
    hydroData, numPrior = AssignHydro(hydroRaw, rainfall.precipRecord)  # interpolate hydrograph data over months
    rainfall.FitHydrograph(hydroData, params)                           # find optimal rainfall to match hydrograph
    rainfall.LabelPrecip(numPrior)                                      # add special index to precip data set

    # run model with posited mean ppt
    rainfall.precipRecord['deficit'] = cumsum(rainfall.precipRecord['precipitation'] - params.pptSelect)
    hydroData, X = rainfall.DistributeDeficit(hydroData, params)
    
    # split into training and validation data sets
    trainData = hydroData[(hydroData['date']>=params.startTrainDate) & (hydroData['date']<=params.splitDate)].copy()
    validateData = hydroData[hydroData['date']>params.splitDate].copy()
    pptDefTrain = trainData[X].values
    pptDefValid = validateData[X].values

    # fits and predictions
    regList = ['linear', 'MVLM', 'random forest', 'SVR', 'ANN']
    regress = Regressions(trainData, regList, params.pptSelect)
    trainData = regress.Train(trainData, pptDefTrain)
    validateData = regress.Validate(validateData, pptDefValid)

    # plot residuals histograms and model-vs-data scatter plots
    PlotResHistograms(trainData, validateData, regList, params.pptSelect)
    PlotScatter(trainData, validateData, regList, params.pptSelect)
 
    # write to summary files
    trainData.to_csv('training.csv')
    validateData.to_csv('validate.csv')
    
    print('Done.')
    
    
### run script ###
HydroPrecip()





