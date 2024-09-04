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
