# 数模代码汇总
## 分类
### 无监督分类
#### k-means聚类
```python
import numpy as np  
from sklearn.datasets import make_blobs  
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import KMeans  
import matplotlib.pyplot as plt  

# 生成模拟数据
X, y_true = make_blobs(n_samples=300, centers=5, cluster_std=0.50, random_state=12345)  

# 数据标准化  
scaler = StandardScaler()  
x = scaler.fit_transform(X)  

# 肘部法则确定K值  
SSE = []  # Store the sum of squared errors for each k  
for k in range(1, 10):  
    estimator = KMeans(n_clusters=k, random_state=0)  # Construct the clustering estimator  
    estimator.fit(x)  # Perform k-means clustering  
    SSE.append(estimator.inertia_)  
plt.figure(figsize=(8, 4))  
plt.plot(range(1, 10), SSE, 'o-')  
plt.xlabel('Number of clusters (k)')  
plt.ylabel('SSE')  
plt.title('Elbow Method Showing Optimal k')  
plt.grid(True)  
plt.show()  

# 使用确定的K值进行聚类
optimal_k = 3  #肘部法得出的k值
kmeans = KMeans(n_clusters=optimal_k, random_state=0)  
kmeans.fit(x)  
labels = kmeans.labels_  
centroids = kmeans.cluster_centers_  

# 显示图形
plt.figure(figsize=(8, 6))  
plt.scatter(x[:, 0], x[:, 1], c=labels, s=50, cmap='viridis', alpha=0.6, edgecolor='w')  
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')  # Centroids  
plt.title('Clusters and Centroids')  
plt.xlabel('Feature 1 (Standardized)')  
plt.ylabel('Feature 2 (Standardized)')  
plt.grid(True)  
plt.show()
```
## 预测
### 时间序列预测
#### LSTM
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


#读取数据
data1 = pd.read_csv('分类日销量统计结果.csv')

#确保时间列是datetime类型
data1['销售日期'] = pd.to_datetime(data1['销售日期'])

# 提取年和月
data1['销售日期'] = data1['销售日期'].dt.to_period('D')

# 对数据进行分组并提取名称为"水仙花"的记录
data2 = data1[data1['分类名称'] == '茄类']

# 对"水仙花"的数据进行分组
data = data2.groupby(['分类名称', '销售日期'])['销量(千克)'].sum().reset_index()

#转换为适合LSTM训练输入
def create_dataset(dataset, steps):
    dataX, dataY = [], []
    for i in range(len(dataset) - steps):
        dataX.append(dataset[i:(i + steps), 0:data.shape[1]])
        dataY.append(dataset[i + steps, 0:data.shape[1]])
    return np.array(dataX), np.array(dataY)


df = pd.DataFrame(data)
df = df[['销售日期', '销量(千克)']]
df.set_index('销售日期', inplace=True)
#将销售额数据转换为numpy数组
dataset = df.values.astype('float32')

#归一化数据到0~1之间
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

#划分训练集和测试集
steps = 30  #设置时间步长
train_size = int(len(dataset)*0.8)  #划分训练集
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[(train_size - steps):len(dataset), :]

# 创建训练集和测试集

trainX, trainY = create_dataset(train, steps)
testX, testY = create_dataset(test, steps)


#创建LSTM模型
model = Sequential()
model.add(LSTM(10, input_shape=(trainX.shape[1], trainX.shape[2])))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=300, batch_size=4)

#预测
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
#计算得分
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))
#计算R方
train_r2 = r2_score(trainY, trainPredict)
test_r2 = r2_score(testY, testPredict)
#反归一化
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY.reshape(-1, 1))
print('Train R2: %.2f' % train_r2)
print('Test R2: %.2f' % test_r2)


#绘图
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[steps:len(trainPredict) + steps, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan

#确保testPredict插入的位置是合理的
test_start = len(trainPredict) + steps
test_end = test_start + len(testPredict)

#输出尺寸调试信息
print("TestPredict shape:", testPredict.shape)
print("Expected slice size:", test_end - test_start)

#确保插入的部分与testPredict尺寸一致
testPredictPlot[test_start:test_end, :] = testPredict

plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.title('预测与真实销售额')
plt.xlabel('时间')
plt.ylabel('销售额（千克）')
plt.show()
```
#### ARIMA
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data = pd.read_csv("分组后的茄类销量数据.csv", parse_dates=['Date'], index_col='Date')
#data = data.asfreq('D')  #设置频率信息，以日计算
data1 = data.loc['2020/7/01':'2023/6/30', '销量(千克)']
# 选取索引值从'2014-01'到'2014-06'的所有行，以及列名为'Close'的列
data1.head()
train = data1.loc['2020/7/01':'2023/6/01']
test = data1.loc['2023/6/02':'2023/6/30']

# 查看训练集的时间序列与数据（只包含训练集）
plt.figure(figsize=(12, 6))
plt.plot(train)
plt.xticks(rotation=45)  # 旋转45度
plt.show()

#差分法
# .diff(1)做一个时间间隔
data['diff_1'] = data['销量(千克)'].diff(1)  # 1阶差分

# 对一阶差分数据再划分时间间隔
data['diff_2'] = data['diff_1'].diff(1)  # 2阶差分

fig = plt.figure(figsize=(12, 10))
# 原数据
ax1 = fig.add_subplot(311)
# add_subplot方法用于在这个图形窗口中添加一个子图网格
# 311用于指定子图的布局,3表示行，1表示列，1表示位置编号
ax1.plot(data['销量(千克)'])
# 1阶差分
ax2 = fig.add_subplot(312)
ax2.plot(data['diff_1'])
# 2阶差分
ax3 = fig.add_subplot(313)
ax3.plot(data['diff_2'])
plt.show()


#ADF检验
import statsmodels.api as sm
# statsmodels提供了许多用于估计和检验统计模型的类和函数，以及进行统计测试的数据和结果的可视化。
from statsmodels.tsa.seasonal import seasonal_decompose
# seasonal_decompose函数用于时间序列的季节性分解，将时间序列数据分解为趋势（trend）、季节性（seasonal）和残差（residual）成分。
from statsmodels.tsa.stattools import adfuller as ADF

# 计算原始序列、一阶差分序列、二阶差分序列的单位根检验结果
data['diff_1'] = data['diff_1'].fillna(0)  # .fillna(0)：填充缺失值
data['diff_2'] = data['diff_2'].fillna(0)

timeseries_adf = ADF(data['销量(千克)'].tolist())  # .tolist()：将ChinaBank['Close']转换为一个Python列表
timeseries_diff1_adf = ADF(data['diff_1'].tolist())
timeseries_diff2_adf = ADF(data['diff_2'].tolist())

# 打印单位根检验结果
print('timeseries_adf：', timeseries_adf)
print('timeseries_diff1_adf：', timeseries_diff1_adf)
print('timeseries_diff2_adf：', timeseries_diff2_adf)

# 绘制
fig = plt.figure(figsize=(12, 7))

ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train, lags=20, ax=ax1)
# lags=20: 这个参数指定了要计算并绘制自相关性的滞后阶数。它会计算并显示从0阶（即当前值与自身的相关性，总是1）到20阶的自相关性
# ax=ax1: 这个参数允许你将绘制的图形添加到指定的Matplotlib轴（axis）对象上。
ax1.xaxis.set_ticks_position('bottom')  # 设置坐标轴上的数字显示的位置
# fig.tight_layout()

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')  # 设置坐标轴上的数字显示的位置
# fig.tight_layout()
plt.show()


# 通过AIC和BIC来求p,q最优值
# statsmodels库中的tsa.arma_order_select_ic函数基于AIC和BIC选择自回归移动平均（ARMA）模型的最佳p（自回归项的阶数）和q（移动平均项的阶数）值
train_results = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='n', max_ar=5, max_ma=5)
# train: 要拟合ARMA模型的时间序列数据
# ic=['aic', 'bic']: 指定使用的信息准则列表
# trend='n': 指定模型中是否包含趋势项。'n'表示不包含趋势项（即只考虑ARMA模型，不考虑趋势）。
# max_ar=8,max_ma=8:最大p为5和最大q为5
print('AIC', train_results.aic_min_order)
print('BIC', train_results.bic_min_order)

# 根据以上求得
p = 3
d = 1
q = 4

# 创建一个ARIMA模型实例
model = sm.tsa.ARIMA(data1, order=(p, d, q))
results = model.fit()
resid = results.resid  # 获取残差

# 绘制残差ACF图
# 查看所选取的整个数据的时间序列与数据
fig, ax = plt.subplots(figsize=(12, 5))

ax = sm.graphics.tsa.plot_acf(resid, lags=40, ax=ax)

plt.show()

#对未来的预测
predict_sunspots=results.predict(dynamic=False)
# results.predict(...): 模型拟合结果对象results的一个方法，用于生成未来时间点的预测值。
# dynamic=False:预测是基于历史数据（即直到最后一个观测值为止的所有数据）进行的，而不是基于之前步骤的预测结果。
print(predict_sunspots)  #检验模型在过去时间点的拟合质量

#MSE
mse = mean_squared_error(data1, predict_sunspots)
print(f'Mean Squared Error: {mse}')
#RMSE
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')
#R方
r2 = r2_score(data1, predict_sunspots)
print(f'R²: {r2}')

# 查看时间序列与数据（蓝色代表原始数据，黄色代表预测数据）
plt.figure(figsize=(12,6))
plt.plot(data1)
plt.xticks(rotation=45) # 旋转45度
plt.plot(predict_sunspots)
plt.show()

# 设置未来预测的步数
forecast_steps = 10

# 获取训练和测试数据的结束点，预测从此点开始
last_train_date = data1.index[-1]

# 使用模型生成未来10天的预测
forecast = results.forecast(steps=forecast_steps)

# 打印预测结果
print(forecast)

# 可视化
plt.figure(figsize=(12,6))
plt.plot(data1, label='Historical Data')
plt.plot(pd.date_range(last_train_date, periods=forecast_steps+1, inclusive='right'), forecast, label='Forecast', color='orange')
plt.xticks(rotation=45) # 旋转45度
plt.legend()
plt.show()


```
#### SARIMA
```python

import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

data = pd.read_csv("分组后的水生根茎类销量数据.csv", parse_dates=['Date'], index_col='Date')
data = data.asfreq('D')  #设置频率信息，以日计算
data1 = data.loc['2020/7/01':'2023/6/30', '销量(千克)']
# 选取索引值从'2014-01'到'2014-06'的所有行，以及列名为'Close'的列
data1.head()
train = data1.loc['2020/7/01':'2023/6/01']
test = data1.loc['2023/6/02':'2023/6/30']

# 查看训练集的时间序列与数据（只包含训练集）
plt.figure(figsize=(12, 6))
plt.plot(train)
plt.xticks(rotation=45)  # 旋转45度
plt.show()

#分解时序
#STL（Seasonal and Trend decomposition using Loess）是一个非常通用和稳健强硬的分解时间序列的方法
import statsmodels.api as sm
#decompostion=tsa.STL(NGE).fit()报错，这里前面加上索引sm
decompostion = sm.tsa.STL(data1, period = None).fit()#statsmodels.tsa.api:时间序列模型和方法
decompostion.plot()
#趋势效益
trend = decompostion.trend
#季节效应
seasonal = decompostion.seasonal
#随机效应
residual = decompostion.resid


#差分法
# .diff(1)做一个时间间隔
data['diff_1'] = data['销量(千克)'].diff(1)  # 1阶差分

# 对一阶差分数据再划分时间间隔
data['diff_2'] = data['diff_1'].diff(1)  # 2阶差分

fig = plt.figure(figsize=(12, 10))
# 原数据
ax1 = fig.add_subplot(311)
# add_subplot方法用于在这个图形窗口中添加一个子图网格
# 311用于指定子图的布局,3表示行，1表示列，1表示位置编号
ax1.plot(data['销量(千克)'])
# 1阶差分
ax2 = fig.add_subplot(312)
ax2.plot(data['diff_1'])
# 2阶差分
ax3 = fig.add_subplot(313)
ax3.plot(data['diff_2'])
plt.show()


#ADF检验
# statsmodels提供了许多用于估计和检验统计模型的类和函数，以及进行统计测试的数据和结果的可视化。
from statsmodels.tsa.seasonal import seasonal_decompose
# seasonal_decompose函数用于时间序列的季节性分解，将时间序列数据分解为趋势（trend）、季节性（seasonal）和残差（residual）成分。
from statsmodels.tsa.stattools import adfuller as ADF

# 计算原始序列、一阶差分序列、二阶差分序列的单位根检验结果
data['diff_1'] = data['diff_1'].fillna(0)  # .fillna(0)：填充缺失值
data['diff_2'] = data['diff_2'].fillna(0)

timeseries_adf = ADF(data['销量(千克)'].tolist())  # .tolist()：将ChinaBank['Close']转换为一个Python列表
timeseries_diff1_adf = ADF(data['diff_1'].tolist())
timeseries_diff2_adf = ADF(data['diff_2'].tolist())

# 打印单位根检验结果
print('timeseries_adf：', timeseries_adf)
print('timeseries_diff1_adf：', timeseries_diff1_adf)
print('timeseries_diff2_adf：', timeseries_diff2_adf)

# 绘制
fig = plt.figure(figsize=(12, 7))

ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train, lags=20, ax=ax1)
# lags=20: 这个参数指定了要计算并绘制自相关性的滞后阶数。它会计算并显示从0阶（即当前值与自身的相关性，总是1）到20阶的自相关性
# ax=ax1: 这个参数允许你将绘制的图形添加到指定的Matplotlib轴（axis）对象上。
ax1.xaxis.set_ticks_position('bottom')  # 设置坐标轴上的数字显示的位置
# fig.tight_layout()

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')  # 设置坐标轴上的数字显示的位置
# fig.tight_layout()
plt.show()

#确定SARIMA模型参数

# 定义参数的范围
p = d = q = range(0, 3)  # 非季节性p,d,q的范围
P = D = Q = range(0, 2)  # 季节性P,D,Q的范围
s = [12]  # 季节周期，可以尝试不同的周期，对于月份如12个月或对于日，7天等

# 创建参数组合
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], s[0]) for x in list(itertools.product(P, D, Q))]

# 网格搜索寻找最优参数
best_aic = np.inf
best_params = None
best_seasonal_params = None

for param in pdq:
    for seasonal_param in seasonal_pdq:
        try:
            model = sm.tsa.statespace.SARIMAX(data1,
                                              order=param,
                                              seasonal_order=seasonal_param,
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)
            results = model.fit()

            if results.aic < best_aic:
                best_aic = results.aic
                best_params = param
                best_seasonal_params = seasonal_param
            print(f'SARIMA{param}x{seasonal_param}12 - AIC:{results.aic}')
        except:
            continue

print(f'Best SARIMA parameters: {best_params} x {best_seasonal_params}12 - AIC:{best_aic}')


# 根据以上求得
p = 3
d = 1
q = 4
P = 2
D = 3
Q = 1
s = 12  # 季节周期为12（假设月度数据）

# 创建SARIMA模型
model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s), enforce_stationarity=False, enforce_invertibility=False)
results = model.fit()
print(results.summary())

# 白噪声检验
resid = results.resid
fig, ax = plt.subplots(figsize=(12, 5))
plot_acf(resid, lags=40, ax=ax)
plt.show()

# Ljung-Box检验
lb_test = sm.stats.acorr_ljungbox(resid, lags=10)
print('Ljung-Box检验结果：', lb_test)

# 预测
predict_sunspots = results.predict(dynamic=False)
print(predict_sunspots)

# MSE, RMSE, R²
mse = mean_squared_error(data1, predict_sunspots)
rmse = np.sqrt(mse)
r2 = r2_score(data1, predict_sunspots)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R²: {r2}')

# 可视化拟合结果
plt.figure(figsize=(12,6))
plt.plot(data1, label='Original Data')
plt.plot(predict_sunspots, label='Predicted Data', color='orange')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# 未来预测
forecast_steps = 10
last_train_date = data1.index[-1]
forecast = results.forecast(steps=forecast_steps)
print(forecast)

# 可视化预测结果
plt.figure(figsize=(12,6))
plt.plot(data1, label='Historical Data')
plt.plot(pd.date_range(last_train_date, periods=forecast_steps+1, inclusive='right'), forecast, label='Forecast', color='orange')
plt.xticks(rotation=45)
plt.legend()
plt.show()
```
## 相关度分析
### perason相关系数
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 从Excel文件中读取数据
file_path = 'your_file.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'  # 替换为你的工作表名称
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 计算皮尔逊相关系数矩阵
correlation_matrix = data.corr(method='pearson')

# 设置绘图风格
sns.set(style='white')

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Pearson Correlation Coefficient Heatmap')
plt.show()
```
### spearman相关系数
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 从Excel文件中读取数据
file_path = 'your_file.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'  # 替换为你的工作表名称
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 计算皮尔逊相关系数矩阵
correlation_matrix = data.corr(method='spearman')

# 设置绘图风格
sns.set(style='white')

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Pearson Correlation Coefficient Heatmap')
plt.show()

```
## 评价决策类
### RSR熵权法TOPSIS
```python
import numpy as np
import pandas as pd


def read(file, sheet_name=0):
    # 使用pandas读取Excel文件
    df = pd.read_excel(file, sheet_name=sheet_name)
    # 将DataFrame转换为NumPy数组
    data_array = df.to_numpy()
    return data_array


def entropy(data0):
    # 返回每个样本的指数
    # 样本数，指标个数
    n, m = np.shape(data0)
    # 一列一个样本，一行一个指标
    # 下面是归一化
    maxium = np.max(data0, axis=1)
    minium = np.min(data0, axis=1)
    data1 = data0.copy()
    for i in range(0, 9):  #自定义指标数
        data0[i] = (data0[i] - minium[i]) * 1.0 / (maxium[i] - minium[i])
    ##计算第j项指标，第i个样本占该指标的比重
    sumzb = np.sum(data0, axis=1)
    for i in range(0, 9):  #自定义指标数
        data0[i] = data0[i] / sumzb[i]
    # 对ln0处理
    a = data0 * 1.0
    a[np.where(data0 == 0)] = 0.0001
    #    #计算每个指标的熵
    e = (-1.0 / np.log(m)) * np.sum(data0 * np.log(a), axis=1)
    #    #计算权重
    w = (1 - e) / np.sum(1 - e)
    print(w)
    return w


# 极小型指标 -> 极大型指标
def dataDirection_1(datas):
    return (np.max(datas) - datas)  # 套公式


# 中间型指标 -> 极大型指标
def dataDirection_2(datas, x_best):
    temp_datas = datas - x_best
    M = np.max(abs(temp_datas))
    answer_datas = 1 - abs(datas - x_best) / M  # 套公式
    return answer_datas


# 区间型指标 -> 极大型指标
def dataDirection_3(datas, x_min, x_max):
    M = max(x_min - np.min(datas), np.max(datas) - x_max)
    answer_list = []
    for i in datas:
        if (i < x_min):
            answer_list.append(1 - (x_min - i) / M)  # 套公式
        elif (x_min <= i <= x_max):
            answer_list.append(1)
        else:
            answer_list.append(1 - (i - x_max) / M)
    return np.array(answer_list)


def temp2(datas):
    K = np.power(np.sum(pow(datas, 2), axis=1), 0.5)
    for i in range(0, K.size):
        for j in range(0, datas[i].size):
            datas[i, j] = datas[i, j] / K[i]  # 套用矩阵标准化的公式
    return datas

#topsis 函数
def topsis(answer, w):
    list_max = []
    for i in answer:
        list_max.append(np.max(i[:]))  # 获取每一列的最大值
    list_max = np.array(list_max)
    list_min = []
    for i in answer:
        list_min.append(np.min(i[:]))  # 获取每一列的最小值
    list_min = np.array(list_min)
    max_list = []  # 存放第i个评价对象与最大值的距离
    min_list = []  # 存放第i个评价对象与最小值的距离
    answer_list = []  # 存放评价对象的未归一化得分
    for k in range(0, np.size(answer, axis=1)):  # 遍历每一列数据
        max_sum = 0
        min_sum = 0
        for q in range(0, 9):  # 自定义指标数
            max_sum += w[q] * np.power(answer[q, k] - list_max[q], 2)  # 按每一列计算Di+
            min_sum += w[q] * np.power(answer[q, k] - list_min[q], 2)  # 按每一列计算Di-
        max_list.append(pow(max_sum, 0.5))
        min_list.append(pow(min_sum, 0.5))
        answer_list.append(min_list[k] / (min_list[k] + max_list[k]))  # 套用计算得分的公式 Si = (Di-) / ((Di+) +(Di-))
        max_sum = 0
        min_sum = 0
    answer = np.array(answer_list)  # 得分归一化
    return (answer / np.sum(answer))

#RSR函数
def RSR(a, w):
    # 获取评价矩阵的对象数和指标数
    num_objects, num_criteria = a.shape

    # 初始化一个矩阵，用于存储排名
    ranks = np.zeros_like(a)

    # 对每个指标进行排名
    for j in range(num_criteria):
        # 使用argsort获取排序后的索引
        ranked_indices = np.argsort(a[:, j])
        # 给每个对象分配排名（从1开始）
        ranks[ranked_indices, j] = np.arange(1, num_objects + 1)

        # 计算每个对象的秩和比（RSR）
    rsr_values = np.sum(ranks, axis=1) / num_criteria

    # 计算每个对象的加权RSR得分
    rsr_scores = np.dot(rsr_values, w)

    return rsr_scores
def main():
    file = 'D:\\kong\\数学建模\\数据汇总.xlsx'
    answer1 = read(file)  # 读取文件
    answer2 = answer1.copy()
    for i in range(0, 9):  # 按照不同的列，根据不同的指标转换为极大型指标（）range自定义
        if i == 3:  # 中间型指标列数（i自定义）
            answer1[i] = dataDirection_2(answer1[i], 0)
        elif i == 1 or i == 2 or i == 8:  # 极小型指标列数（i自定义）
            answer1[i] = dataDirection_1(answer1[i])
        elif i == 6:  #区间型指标列数（i自定义）
            answer1[i] = dataDirection_3(answer1[i],x_min=0,x_max=5)
        else:  # 本来就是极大型指标，不用转换
            answer1[i] = answer1[i]
    answer3 = temp2(answer1)  # 正向数组标准化
    w = entropy(answer2)  # 计算权重
    answer4 = topsis(answer3, w)  # topsis使用
    data = pd.DataFrame(answer4)  # 计算得分
    print(data)
    # 将得分输出到excel表格中
    writer = pd.ExcelWriter('D:\\kong\\数学建模\\结果.xlsx')  # 写入Excel文件
    data.to_excel(writer, '得分', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    writer.save()
    writer.close()


if __name__ == '__main__':
    main()
```
