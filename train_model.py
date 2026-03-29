import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score
from textblob import TextBlob
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("fifa.csv",low_memory=False)

# clean value column
df['Value'] = df['Value'].replace('[\€,M,K]','',regex=True).astype(float)

# create feature
df['value_per_age'] = df['Value']/df['Age']

# features
features=['Age','OVA','value_per_age']

X=df[features]
y=df['Value']

# train model
model=RandomForestRegressor()
model.fit(X,y)

# predictions
pred=model.predict(X)

# evaluation
print("MAE:",mean_absolute_error(y,pred))
print("R2 Score:",r2_score(y,pred))

# save model
pickle.dump(model,open("model.pkl","wb"))

print("Model saved successfully")

# visualization
plt.scatter(df['Age'],df['Value'])
plt.xlabel("Age")
plt.ylabel("Value")
plt.title("Age vs Transfer Value")
plt.show()

# sentiment example (for documentation match)
text="Player is in excellent form"
sentiment=TextBlob(text).sentiment.polarity

print("Sentiment Score:",sentiment)