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