# Renthop-challenge
## Problem Description: 
In this competition, you will predict how popular an apartment rental listing is based on the listing content like text description, photos, number of bedrooms, price, etc. The data comes from renthop.com, an apartment listing website. These apartments are located in New York City. The target variable, interest_level, is defined by the number of inquiries a listing has in the duration that the listing was live on the site.

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

![13](https://github.com/akshaygade/Renthop-/blob/master/images/13.png)


### Clustering using K-means

![13](https://github.com/akshaygade/Renthop-/blob/master/images/13.png)





