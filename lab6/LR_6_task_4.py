from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('ave_data.csv')
df = df.dropna()


max_val = round(df['price'].max()) + 10
bins = range(0, max_val, 10)
labels = [f"{i}-{i+10}" for i in range(0, max_val-10, 10)]
df['price_range'] = pd.cut(df['price'], bins=bins, right=False, labels=labels)
y = df['price_range']

label_encoders = {}
for column in ['origin', 'destination', 'train_type', 'train_class', 'fare']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

df['start_day_of_week'] = pd.to_datetime(df['start_date']).dt.dayofweek
df['duration'] = pd.to_datetime(df['end_date']) - pd.to_datetime(df['start_date'])
df['duration_minutes'] = (df['duration'].dt.total_seconds()//60).astype(int)

# df['price'] = (df['price'] * 100).astype(int)


X = df[['origin', 'destination', 'start_day_of_week', 'duration_minutes', 'train_type', 'train_class', 'fare']]
# y = df['price']
y = df['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=5)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=np.nan, ))

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()