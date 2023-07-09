import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
import pickle

df = pd.read_csv("weather_data.csv")

df = df.drop(columns=["LandAvgTempUncertainty", "LandMaxTempUncertainty",
                      "LandMinTempUncertainty", "Land_OceanAvgTempUncertainty"], axis=1)

# Converting the date column to datetime object
df["dt"] = pd.to_datetime(df["dt"])
df["Month"] = df["dt"].dt.month
df["Year"] = df["dt"].dt.year

# Removing month and dt column since we are predicting data based on annual record
df = df.drop(["dt", "Month"], axis=1)
df = df[df.Year >= 1850]
df = df.set_index(["Year"])
df = df.dropna()

# Correlation Matrix
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

# Splitting dependent and independent variables
y = df["Land_OceanAvgTemp"]
x = df[["LandAvgTemp", "LandMaxTemp", "LandMinTemp"]]

# Train-test split for training the model
xtrain, x_test, ytrain, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = make_pipeline(
    SelectKBest(k="all"),
    StandardScaler(),
    RandomForestRegressor(n_estimators=100, max_depth=50, random_state=77, n_jobs=-1)
)

model.fit(xtrain, ytrain)

y_pred = model.predict(x_test)
model.predict([[0.749, 8.242, -3.206]])

pickle_out = open("classifier.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

