import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import metrics
from math import sqrt
import openpyxl
import scipy
from openpyxl import load_workbook
import math
from statsmodels.tsa.seasonal import seasonal_decompose
from numpy.polynomial import Polynomial

df = pd.read_excel('C:/Users/Пользователь/PycharmProjects/raschet/usa_gas.xlsx')
format = ['mon', 'value']
df_selected = df[format] # датафрэйм для добычи газа в США с 1971 по 1990


def plot_df(df_selected, x, y, title="", style='bmh'):
    with plt.style.context(style):
        df_selected.plot(x, y, figsize=(16, 5))
        plt.gca().set(title=title)
        # plt.show()

# plot_df(df_selected, "mon", "value",
#         title='Monthly gas production (billion cubic feet) in USA from 1975 to 1984')

t = list(range(1, len(df_selected) + 1))

def t_2(x):
    s = []
    for i in list(range(1, len(x) + 1)):
        s.append(i ** 2)
    return s

t = pd.DataFrame(t)
t_sq = pd.DataFrame(t_2(df))

col_t = list(t.columns)
col_t_sq = list(t_sq.columns)

col_t[0] = 't'
col_t_sq[0] = 't^2'

t.columns = col_t
t_sq.columns = col_t_sq

t.set_index(df_selected.index, inplace = True)
t_sq.set_index(df_selected.index, inplace = True)

df = pd.concat([df_selected, t, t_sq], axis=1) #выводим нормальную таблицу с t и t**2
# print(df)

## Выделим тренд, сезонность и остатки с помощью STL
# Мультипликативное разложение
result_mul = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq', period=12)

# Аддитивное разложение
result_add = seasonal_decompose(df['value'], model='additive', extrapolate_trend='freq', period=12)


#Plot
plt.rcParams.update({'figure.figsize': (20, 20)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize = 22)
result_add.plot().suptitle('Additive Decompose', fontsize = 22)
# plt.show()

df_reconstructed = pd.concat([result_mul.seasonal, result_mul.trend, result_mul.resid, result_mul.observed], axis=1)
df_reconstructed.columns = ['seasonal', 'trend', 'residuals', 'actual_values']
df_reconstructed.head()
# print(df_reconstructed)

def period(df):
    a, b = [], []
    a.append(df.mean()) # Alpha_0
    b.append(0)

    for j in range(1, len(df)//2):
        p = 0
        q = 0
        for t in range(1, len(df) + 1):
            p = p + df[t-1] * math.cos(2 * math.pi * j * t / len(df))
            q = q + df[t-1] * math.sin(2 * math.pi * j * t / len(df))

        a.append(2 / len(df) * p) # коэффы Alpha_j (всего их T/2)
        b.append(2 / len(df) * q) # коэффы Beta_j (всего их T/2)


    T_2 = 0
    for t in range(1, len(df) + 1):
        T_2 += 1/len(df) * (-1)**t * df[t-1]

    a.append(T_2)
    b.append(0)

    periodogramma = []

    for i in range(len(a)):
        I_j = (a[i] ** 2 + b[i] ** 2) * len(df) // 2 #интенсивность для j-й гармоники
    periodogramma.append(I_j)

    fig, ax = plt.subplots()
    plt.plot(range(len(periodogramma)), periodogramma, c='Blue')
    ax.legend(loc = 'upper center', fontsize = 25, ncol = 2)
    fig.set_figwidth(20)
    fig.set_figheight(6)
    fig.suptitle('Периодограмма')
    plt.show()

    # Разложение исходного ряда в ряд Фурье
    Furie = []
    for t in range(1, len(df) + 1):
        x = 0
        for j in range(len(a)):
            x = x + a[j] * math.cos(2 * math.pi * j * t/len(df)) + b[j] * math.sin(2 * math.pi * j * t/len(df))
        Furie.append(x)
    return periodogramma
#return a, b, periodogramma, Furie
# y1 = list(fluctuations)
# y2 = period(fluctuations)[3]
# fig, ax = plt.subplots()
# plt.plot(range(len(y1)), y1, c = 'hotpink', linewidth = 4, label = 'Data')
# plt.plot(range(len(y2)), y2, c = 'Black', linewidth = 3, label = 'Forecast')
# ax.legend(loc = 'upper center', fontsize = 25, ncol = 2)
# fig.set_figwidth(20)
# fig.set_figheight(6)
# fig.suptitle("Представление в виде конечного ряда Фурье")
# plt.show()
# print(period(fluctuations))


## Модель линейного тренда
print('ЛИНЕЙНЫЙ ТРЕНД')
x2 = df[['t']]
y2 = df['value']
estimator2 = LinearRegression()
estimator2.fit(x2, y2)
y_pred2 = estimator2.predict(x2)
# print(f'Slope : {format(round(estimator2.coef_[0],2))}') #a1
# print(f'Intercept : {format(round(estimator2.intercept_,2))}') #const
# print(f'R^2 : {round(estimator2.score(x2,y2),2)}')

model_s = smf.ols('value ~ t', data=df)
res1 = model_s.fit()
# print(res1.summary())

## Квадратичный тренд
print("КВАДРАТИЧНЫЙ ТРЕНД")
y1 = df['value']
x1 = df[['t', 't^2']]
estimator1 = LinearRegression()
estimator1.fit(x1, y1)
y_pred1 = estimator1.predict(x1)
#print(f'Slope 1 : {format(round(estimator1.coef_[0],2))}') #угол наклона перед t, a1
#print(f'Slope 2 : {format(round(estimator1.coef_[1],4))}') #угол наклона перед t^2, a2
#print(f'Intercept : {format(round(estimator1.intercept_,2))}') #константа
#print(f'R^2 : {round(estimator1.score(x1,y1),2)}')

model_sq = smf.ols('value ~ t + t^2', data=df)
res2 = model_sq.fit()
# print(res2.summary())


## Сезонные переменные, все месяцы за исключением Июля
def mon(id, name):
    y = []
    for i in list(range(len(df.index))):
        if i % 12 == id:
            y.append(1)
        else:
            y.append(0)
    y = pd.DataFrame(y)
    y.set_index(df.index, inplace = True)
    # month = y

    col_month = list(y.columns)

    col_month[0] = name

    y.columns = col_month
    return y


months = {
    'January': 0, 'February': 1, 'March': 2,
    'April': 3, 'May': 4, 'June': 5,
    'August': 7, 'September': 8, 'October': 9,
    'November': 10, 'December': 11
}

df = pd.concat([df, mon(months["January"], "January"), mon(months["February"], 'February'),
                mon(months["March"], 'March'), mon(months['April'], 'April'),  mon(months['May'], 'May'),
                mon(months['June'], 'June'), mon(months["August"], 'August'), mon(months["September"], 'September'),
                mon(months["October"], 'October'), mon(months["November"], 'November'),
                mon(months["December"], 'December')], axis=1)
# print(df)


##  Линейный тренд с сезонными фиктивными переменными
x4 = df[['t', 'January', 'February', 'March', 'April',
         'May', 'June', 'August', 'September', 'October', 'November', 'December']]
y4 = df['value']
estimator4 = LinearRegression()
estimator4.fit(x4, y4)
y_pred4 = estimator4.predict(x4)

#print(f'Slope 1 : {format(round(estimator4.coef_[0],2))}') #a1
#print(f'Slope 2 : {format(round(estimator4.coef_[1],4))}') #a2 January
#print(f'Slope 3 : {format(round(estimator4.coef_[2],4))}') #a3 February
#print(f'Slope 4 : {format(round(estimator4.coef_[3],4))}') #a4 March
#print(f'Slope 5 : {format(round(estimator4.coef_[4],4))}') #a5 May
#print(f'Slope 6 : {format(round(estimator4.coef_[5],4))}') #a6 June
#print(f'Slope 7 : {format(round(estimator4.coef_[6],4))}') #a7 July
#print(f'Slope 8 : {format(round(estimator4.coef_[7],4))}') #a8 August
#print(f'Slope 9 : {format(round(estimator4.coef_[8],4))}') #a9 September
#print(f'Slope 10 : {format(round(estimator4.coef_[9],4))}') #a10 October
#print(f'Slope 11 : {format(round(estimator4.coef_[10],4))}') #a11 November
#print(f'Slope 12 : {format(round(estimator4.coef_[11],4))}') #a12 December
#print(f'Intercept: {format(round(estimator4.intercept_,2))}') #const
#print(f'R^2 : {round(estimator4.score(x4, y4),2)}')

## Квадратичный тренд с сезонными фиктивными переменными
x3 = df[['t', 't^2', 'January', 'February', 'March', 'April',
         'May', 'June', 'August', 'September', 'October', 'November', 'December']]
y3 = df['value']
estimator3 = LinearRegression()
estimator3.fit(x3, y3)
y_pred3 = estimator3.predict(x3)

#print(f'Slope 1: {format(round(estimator3.coef_[0],2))}') #a1
#print(f'Slope 2: {format(round(estimator3.coef_[1],4))}') #a2
#print(f'Slope 3: {format(round(estimator3.coef_[2],4))}') # угол наклона перед January (a3)
#print(f'Slope 4: {format(round(estimator3.coef_[3],4))}') # угол наклона перед February (a4)
#print(f'Slope 5: {format(round(estimator3.coef_[4],4))}') # угол наклона перед March (a5)
#print(f'Slope 6: {format(round(estimator3.coef_[5],4))}') # угол наклона перед May (a6)
#print(f'Slope 7: {format(round(estimator3.coef_[6],4))}') # угол наклона перед June (a7)
#print(f'Slope 8: {format(round(estimator3.coef_[7],4))}') # угол наклона перед July (a8)
#print(f'Slope 9: {format(round(estimator3.coef_[8],4))}') # угол наклона перед August (a9)
#print(f'Slope 10: {format(round(estimator3.coef_[9],4))}') # угол наклона перед September (a10)
#print(f'Slope 11: {format(round(estimator3.coef_[10],4))}') # угол наклона перед October (a11)
#print(f'Slope 12: {format(round(estimator3.coef_[11],4))}') # угол наклона перед November (a12)
#print(f'Slope 13: {format(round(estimator3.coef_[12],4))}') # угол наклона перед December (a13)
#print(f'Intercept : {format(round(estimator3.intercept_,2))}') # константа
#print(f'R^2 : {round(estimator3.score(x3, y3),2)}')

## Графики с трендами
# Линейный тренд
#fig, axs = plt.subplots(2, 2, figsize=(28,20))
#axs[0,0].plot(y_pred2, color='red')
#axs[0,0].plot(list(y2), color='blue')
#axs[0,0].set_title('Linear trend')
#axs[0,0].set_xlabel('mon')
#axs[0,0].set_ylabel('Monthly Gas Production in USA')

#axs[0,1].plot(y_pred1, color='red')
#axs[0,1].plot(list(y1), color='blue')
#axs[0,1].set_title('Quadratic trend')
#axs[0,1].set_xlabel('mon')
#axs[0,1].set_ylabel('Monthly Gas Production in USA')

#axs[1,0].plot(y_pred4, color='red')
#axs[1,0].plot(list(y4), color='blue')
#axs[1,0].set_title('Linear trend with dummy variables')
#axs[1,0].set_xlabel('mon')
#axs[1,0].set_ylabel('Monthly Gas Production in USA')

#axs[1,1].plot(y_pred3, color='red')
#axs[1,1].plot(list(y3), color='blue')
#axs[1,1].set_title('Quadratic trend with dummy variables')
#axs[1,1].set_xlabel('mon')
#axs[1,1].set_ylabel('Monthly Gas Production in USA')

#plt.show()

# Проверим стат значимость моделей
# Для квадратичного тренда с фикт пер
model_sqf = smf.ols('value ~ t + t^2 + January + February + March + April + May + June + August + September + October + November + December', data=df)
res3 = model_sqf.fit()
#print(res3.summary())

# Для линейного тренда с фикт пер
model_lf = smf.ols('value ~ t + January + February + March + April + May + June + August + September + October + November + December', data=df)
res4 = model_lf.fit()
#print(res4.summary())

y = df[['value']]

## Прогнозирование
s = int(len(df)*0.8) # Длина обучающей выборки (80% всех наблюдений)

# Разделим выборки на train/test
Xtrn1 = x1[0:s]
Xtrn2 = x2[0:s]
Xtrn3 = x3[0:s]
Xtrn4 = x4[0:s]

Xtest1 = x1[s:]
Xtest2 = x2[s:]
Xtest3 = x3[s:]
Xtest4 = x4[s:]

Ytest = y[s:]
Ytrn = y[0:s]

############### Точечный прогноз

## Оценка качества модели
## Метрики качества прогноза
def MAPE(y_pred, y_true): # с ошибкой
    n = len(y_pred)
    return 1 / n * sum(abs(y_pred - y_true) / y_true)

def RMSE(y_pred, y_true): ## среднеквадратическое отклонение
    n = len(y_pred)
    return np.sqrt(1 / n * sum((y_pred - y_true) ** 2))

## Создаем временные структуры
TestModels = pd.DataFrame()
tmp = {} # словарь для параметров после обучения
score = [] # список для результатов

x1 = df[['t']] # linear
x2 = df[['t', 't^2']] # quadratic

x3 = df[['t', 'January', 'February', 'March', 'April', 'May', 'June', 'August',
         'September', 'October', 'November', 'December']] # linear + dummy

x4 = df[['t', 't^2', 'January', 'February', 'April', 'March','May', 'June', 'August',
         'September', 'October', 'November', 'December']] # quadr + dummy
y = df[['value']] # целевая переменная

s = int(len(df)*0.8) # Длина обучающей выборки (80% всех наблюдений)

# Разделим выборки на train/test
Xtrn1 = x1[0:s]
Xtrn2 = x2[0:s]
Xtrn3 = x3[0:s]
Xtrn4 = x4[0:s]

Xtest1 = x1[s:]
Xtest2 = x2[s:]
Xtest3 = x3[s:]
Xtest4 = x4[s:]

Ytest = y[s:]
Ytrn = y[0:s]

model = LinearRegression(fit_intercept=True)
trends = ['Linear', 'Quadratic', 'Linear with Dummy', 'Quadratic with Dummy']

for trend in trends:
    if trend == trends[0]:
        tmp['Model'] = trends[0]

        model.fit(Xtrn1, Ytrn)
        y_pred1 = model.predict(Xtest1)
        tmp['RMSE'] = round(RMSE(y_pred1, Ytest.values)[0],2)
        tmp['MAPE'] = round(MAPE(y_pred1, Ytest.values)[0],2)
        tmp['accuracy'] = round(1 - tmp['RMSE']/np.average(Ytest), 2)
        tmp['coefficient'] = round(model.coef_[0][0],4)

    if trend == trends[1]:
        tmp['Model'] = trends[1]
        model.fit(Xtrn2, Ytrn)
        y_pred2 = model.predict(Xtest2)
        tmp['RMSE'] = round(RMSE(y_pred2, Ytest.values)[0],2)
        tmp['MAPE'] = round(MAPE(y_pred2, Ytest.values)[0],2)
        tmp['accuracy'] = round(1 - tmp['RMSE']/np.average(Ytest),2)

        for i in list(range(len(model.coef_[0]))):
            model.coef_[0][i] = round(model.coef_[0][i],4)

        tmp['coefficient'] = model.coef_[0]


    if trend == trends[2]:
       tmp['Model'] = trends[2]
       model.fit(Xtrn3, Ytrn)
       y_pred3 = model.predict(Xtest3)

       tmp['RMSE'] = round(RMSE(y_pred3, Ytest.values)[0],2)
       tmp['MAPE'] = round(MAPE(y_pred3, Ytest.values)[0],2)
       tmp['accuracy'] = round(1 - tmp['RMSE']/np.average(Ytest),2)

       for i in list(range(len(model.coef_[0]))):
           model.coef_[0][i] = round(model.coef_[0][i],4)

       tmp['coefficient'] = model.coef_[0]
       #print('Coefficients for Linear with Dummy: ' + str(tmp['coefficient']))

    if trend == trends[3]:
       tmp['Model'] = trends[3]
       model.fit(Xtrn4, Ytrn)
       y_pred4 = model.predict(Xtest4)

       tmp['RMSE'] = round(RMSE(y_pred4, Ytest.values)[0],2)
       tmp['MAPE'] = round(MAPE(y_pred4, Ytest.values)[0],2)
       tmp['accuracy'] = round(1 - tmp['RMSE']/np.average(Ytest),2)

       for i in list(range(len(model.coef_[0]))):
           model.coef_[0][i] = round(model.coef_[0][i],4)

       tmp['coefficient'] = model.coef_[0]
       #print('Coefficients for Quadratic with Dummy: ' + str(tmp['coefficient']))
    TestModels = TestModels.append([tmp])

#TestModels.set_index('Model', inplace = True)
#print(TestModels)

#print(df)

# Экспоненциальный тренд без фикт пер
x5 = df['t']
y5 = df['value']
fit = np.polyfit(x5, np.log(y5), 1)
fig, ax = plt.subplots()
intercept1 = math.exp(fit[0])
a0 = math.exp(fit[1])
print(intercept1, a0)
#plt.plot(range(len(y1)), y1, c = 'hotpink', linewidth = 4, label = 'Data')
#plt.plot(range(len(y2)), y2, c = 'Black', linewidth = 3, label = 'Forecast')
#ax.legend(loc = 'upper center', fontsize = 25, ncol = 2)
#fig.set_figwidth(20)
#fig.set_figheight(6)
#fig.suptitle("Exponencial trend")
#ax.plot(fit, color='red')
#ax.plot(list(y5), color='blue')
#plt.show()

#axs[1,1].set_title('Quadratic trend with dummy variables')
#axs[1,1].set_xlabel('Year')
#axs[1,1].set_ylabel('Monthly Gas Production in USA')

#print(fit)

# Спирмен
def spirmen():
    teta = []
    sortedvalues = sorted(df['value'].to_list())
    for i in df['value']:
        teta.append(sortedvalues.index(i)+1)
    tlist = df['t'].to_list()
    squared_dif = 0
    for value in range(len(tlist)):
        squared_dif += ((tlist[value]-teta[value])**2)
    etta = 1 - ((6*squared_dif)/(len(tlist)*((len(tlist))**2 - 1)))

    tstat = scipy.stats.t.ppf(0.975, len(tlist)-2)
    if abs(etta)<tstat:
        print('Тренда нет', abs(etta), tstat)
    else:
        print('Тренд есть', abs(etta), tstat)

spirmen()
