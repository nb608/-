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
