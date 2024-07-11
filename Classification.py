import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

df = pd.read_excel('/Users/komalb/MS/Math 448/Project/Final Songs.xlsx')
df.dropna(inplace=True)

features_to_scale = ['Danceability', 'Energy','Loudness','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo']

scaler = StandardScaler()
features_scaled_df = pd.DataFrame(scaler.fit_transform(df[features_to_scale]), columns=features_to_scale, index=df.index)
df[features_to_scale] = features_scaled_df

df["popularity_class"] = df['Popularity'].apply(lambda x: "Popular" if x>=50 else "Not popular")

all_features = features_to_scale+['Mode', 'Key']
target = 'popularity_class'

significant_features = ['Danceability', 'Energy', 'Loudness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence']

X = df[significant_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)

# Initialize and fit the Logistic Regression model with class weight 'balanced'
logistic_model = LogisticRegression(class_weight='balanced', random_state=29)
logistic_model.fit(X_train, y_train)
logistic_pred = logistic_model.predict(X_test)

# Evaluate the Logistic Regression model
logistic_accuracy = accuracy_score(y_test, logistic_pred)
logistic_recall = recall_score(y_test, logistic_pred,pos_label="Popular")
logistic_precision = precision_score(y_test, logistic_pred,pos_label="Popular")

print("Logistic Regression Accuracy:", round(logistic_accuracy,3))
print("Logistic Recall:", round(logistic_recall,3))
print("Logistic Precision:", round(logistic_precision,3))

# Initialize and fit the SVM model with class weight 'balanced'
svm_model = SVC(class_weight='balanced', random_state=29)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# Evaluate the SVM model
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred,pos_label="Popular")
svm_precision = precision_score(y_test, svm_pred,pos_label="Popular")

print("SVM Accuracy:", round(svm_accuracy,3))
print("SVM Recall :", round(svm_recall,3))
print("SVM Precision:", round(svm_precision,3))

## Initialize and fit the SVM model with class weight 'balanced'
rf_model = RandomForestClassifier(n_estimators=400, class_weight='balanced', random_state=29)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluate the SVM model
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred,pos_label="Popular")
rf_precision = precision_score(y_test, rf_pred,pos_label="Popular")

print("Random Forest Accuracy:", round(rf_accuracy,3))
print("Random Forest Recall :", round(rf_recall,3))
print("Random Forest Precision:", round(rf_precision,3))