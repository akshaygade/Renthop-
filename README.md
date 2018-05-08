# Renthop-challenge
## Problem Description: 
Finding the perfect rental property to call your new home in New York can be exhausting. Scrolling through seemingly bottomless rental listings while looking for that extra bedroom for the kids or that spacious living room where one can fit their humongous home theatre can be a behemoth task. Difficult as it may be, imagine having to make sense and structure all the available real estate property listings to better suit the consumer’s needs and preferences. 

We have taken up that task of delivering a system which provides value to its users when it accurately predicts the number of inquiries a rental listing receives based on the number of bedrooms/bathrooms, its location or the number of additional features. The higher the number of inquiries, the greater the interest shown by the renter for that listing, which in turn lets us identify potential quality issues and reduce the time from posting listings to apartment handover. We present the case study of our system, after careful machine learning and neural network modelling, to predict which of the available rental property features best bundle up to bring a listing most likely to grab the average consumers eye.

## Data fields:
bathrooms: number of bathrooms bedrooms: number of bathrooms building_id created description display_address features: a list of features about this apartment latitude listing_id longitude manager_id photos: a list of photo links. You are welcome to download the pictures yourselves from renthop's site, but they are the same as imgs.zip. price: in USD street_address interest_level: this is the target variable. It has 3 categories: 'high', 'medium', 'low'.

## Our Approach:

### Exploratory Data Analysis

![1](https://github.com/akshaygade/Renthop-/blob/master/images/1.png)
![2](https://github.com/akshaygade/Renthop-/blob/master/images/2.png)
![3](https://github.com/akshaygade/Renthop-/blob/master/images/3.png)
![4](https://github.com/akshaygade/Renthop-/blob/master/images/4.png)
![5](https://github.com/akshaygade/Renthop-/blob/master/images/5.png)
![6](https://github.com/akshaygade/Renthop-/blob/master/images/6.png)
![7](https://github.com/akshaygade/Renthop-/blob/master/images/7.png)
![8](https://github.com/akshaygade/Renthop-/blob/master/images/8.png)
![9](https://github.com/akshaygade/Renthop-/blob/master/images/9.png)

### Word Clouds for features and description

```python
import matplotlib as plt
import pandas as pd
data = pd.read_json('data.json')
df = pd.DataFrame(data)
from wordcloud import WordCloud
text = ''
for row in train_df.iterrows():
    #Change this to features for feature's wordcloud
    for feature in row['description']:
         text = " ".join([text, ".".join(feature.strip().split(" "))])
text = text.strip()
#visualization
plt.figure(figsize=(12,8))
wordcloud = WordCloud(background_color='black', width=600, height=300, max_font_size=60, max_words=30).generate(text)
wordcloud.recolor(random_state=0)
plt.imshow(wordcloud)
plt.title("Features", fontsize=30)
plt.axis("off")
plt.show()
```
![11](https://github.com/akshaygade/Renthop-/blob/master/images/11.png)
![12](https://github.com/akshaygade/Renthop-/blob/master/images/12.png)

### Most frequent words in features
```python
features_list=df.features.tolist()
# features_list
words = str(features_list).split()
Counter = Counter(words)

most_occur = Counter.most_common(100)
most_occur[1:30]

```
![10](https://github.com/akshaygade/Renthop-/blob/master/images/10.png)


### Sentiment Analysis

```python
from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
df['tokens'] = df['description'].apply(sent_tokenize)


def sentiment(words):
    polarity = SIA()
    output = []
    for word in words:
        out = polarity.polarity_scores(word)
        output.append(out)
    return pd.DataFrame(output).mean()
    
df2 = df['tokens'].apply(sentiment)
df2.head()

sentiment_data = pd.concat([df2,df['interest_level']],axis=1)
sentiment_data.head()

sns.set(style="darkgrid", palette="pastel", color_codes=True)
sns.boxplot(x="compound", y="interest_level", hue="interest_level", data=sentiment_data);
sns.boxplot(x="neg", y="interest_level", hue="interest_level", data=sentiment_data);
sns.boxplot(x="pos", y="interest_level", hue="interest_level", data=sentiment_data);
sns.boxplot(x="neu", y="interest_level", hue="interest_level", data=sentiment_data);
```
![13](https://github.com/akshaygade/Renthop-/blob/master/images/13.png)



### Clustering using K-means

```python

fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
scatter = ax.scatter(df['longitude'], df['latitude'], c='m', edgecolor='k', alpha=.4, s=150)
plt.show(scatter)

x = df
latmean = x['latitude'].mean()

lonmean = x['longitude'].mean()

x=x.reset_index()

faa=[]
for i in range(0,len(x)):
    faa.append(np.sqrt(((x['latitude'][i] - latmean) * (x['latitude'][i] - latmean) ) + ((x['longitude'][i] - lonmean) * (x['longitude'][i] - lonmean)) ))

x['distance'] = faa
x = x[x['distance'] < 0.5]
fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
scatter = ax.scatter(x['longitude'], x['latitude'], c='m', edgecolor='k', alpha=.4, s=20)
plt.show()


# K Means Cluster
ncomp = 5
km = KMeans(ncomp, random_state=1)
km.fit(x['latitude'].reshape(-1,1))
x['labels'] = km.labels_
:
#Visualizing the clusters
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

style.use('fivethirtyeight')
fig.set_size_inches(10, 6)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 20, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 20, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 40, c = 'yellow', label = 'Centroids')
```
![14](https://github.com/akshaygade/Renthop-/blob/master/images/14.png)


### Feature Engineering
#### Exploring the geographic location of all the listings
#### Plotting the neighborhoods in NYC
![15](https://github.com/akshaygade/Renthop-/blob/master/images/15.png)
#### Plotting the number of data points on top of the layer
![16](https://github.com/akshaygade/Renthop-/blob/master/images/16.png)
### Exploring transportation by calculating nearest subway station
![17](https://github.com/akshaygade/Renthop-/blob/master/images/17.png)


### Machine Learning
Importing packages
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
```

Train-test split
```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4,random_state=42)
```

Standardizing the variables
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
```

#### Multi Layer Perceptron
```python
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(150,150,150))
mlp.fit(X_train,y_train)
pred = mlp.predict(X_val)

fpr, tpr, threshold = metrics.roc_curve(y_val, pred,pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

```

#### Random Forest
```python

clf = RandomForestClassifier(n_estimators=50,n_jobs=-1,min_samples_leaf=5)
%time clf.fit(X_train, y_train)
y_val_pred = clf.predict(X_val)
accuracy_score(y_val,y_val_pred)
```
#### SVM
```python
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set 
y_pred = classifier.predict(X_val)
```

#### Artificial Neural Netwok
``` python  
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

#Initialize neural network
classifier = Sequential()

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [1],
              'epochs': [5],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5)
grid_search = grid_search.fit(X_train, y_train)


best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
best_accuracy

```


#### K nearest neighbors

```python
KNN_grid = [{'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17], 'weights': ['uniform', 'distance']}]

    # build a grid search to find the best parameters
gridsearchKNN = GridSearchCV(KNeighborsClassifier(), KNN_grid, cv=5)

    # run the grid search
gridsearchKNN.fit(X_train, y_train)
pred=gridsearchKNN.predict(X_val)

```

#### Decision Tree

```python

DT_grid = [{'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'criterion': ['gini', 'entropy']}]

# build a grid search to find the best parameters
gridsearchDT = GridSearchCV(DecisionTreeClassifier(), DT_grid, cv=5)

# run the grid search
gridsearchDT.fit(X_train, y_train)
pred=gridsearchDT.predict(X_val)

```


#### Logistic Regression

``` python

LREG_grid = [{'C': [0.5, 1, 1.5, 2], 'penalty': ['l1', 'l2']}]
gridsearchLREG = GridSearchCV(LogisticRegression(), LREG_grid, cv=5)
gridsearchLREG.fit(X_train, y_train)
pred=gridsearchLREG.predict(X_val)

print(accuracy_score(y_val, pred))

```


### Model results

#### We have considered ROC - AUC curves as our metric and the below picture shows the comparision of all the models, the best model being Multilayer Perceptron
![18](https://github.com/akshaygade/Renthop-/blob/master/images/18.png)

