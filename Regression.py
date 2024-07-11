import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

''' Data Preparation and Pre-processing
                                        '''

df = pd.read_excel('/Users/komalb/MS/Math 448/Project/Final Songs.xlsx')
df.dropna(inplace=True)

features_to_scale = ['Danceability', 'Energy','Loudness','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo']

scaler = StandardScaler()
features_scaled_df = pd.DataFrame(scaler.fit_transform(df[features_to_scale]), columns=features_to_scale, index=df.index)
df[features_to_scale] = features_scaled_df

all_features = features_to_scale + ['Mode', 'Key']
target = 'Popularity'

'''
    Identifying significcant features
                                      '''
X_temp = sm.add_constant(df[all_features])
y = df[target]

ols_model = sm.OLS(y, X_temp).fit()

# Extract p-values from the model
p_values = ols_model.pvalues
p_values = p_values.drop('const')  # Remove the intercept

# Set significance level
alpha = 0.05

# Identify significant features
significant_features = p_values[p_values < alpha].index.tolist()

X = df[significant_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=29)

'''
    Linear Regression
                     '''

lin_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)

lin_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

lin_pred = lin_model.predict(X_test).astype(int)
ridge_pred = ridge_model.predict(X_test).astype(int)

lin_r2 = r2_score(y_test, lin_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

# Since accuracy is not meaningful for regression, we focus on R² scores
print("Linear Regression R²:", round(lin_r2,3))
print("Ridge Regression R²:", round(ridge_r2,3))

'''
    Polynomial Linear Regression
                                '''

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# Split the polynomial features data into training and testing sets
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.12, random_state=29)

# Fit the models with polynomial features
lin_model.fit(X_train_poly, y_train)
ridge_model.fit(X_train_poly, y_train)

# Predict using the models with polynomial features
lin_poly_pred = lin_model.predict(X_test_poly).astype(int)
ridge_poly_pred = ridge_model.predict(X_test_poly).astype(int)

# Evaluate the models with polynomial features
lin_poly_r2 = r2_score(y_test, lin_poly_pred)
ridge_poly_r2 = r2_score(y_test, ridge_poly_pred)

print("Polynomial Linear Regression R²:", round(lin_poly_r2,3))
print("Polynomial Ridge Regression R²:", round(ridge_poly_r2,3))

'''
Random Forest Classifier
                        '''

rf_model = RandomForestRegressor(n_estimators=400, random_state=29)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test).astype(int)

rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest R²:", round(rf_r2,3))