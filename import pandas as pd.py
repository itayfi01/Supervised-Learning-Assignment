import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

sf_lat, sf_lon = 37.7749, -122.4194 # Coordinates for San Francisco
la_lat, la_lon = 34.0522, -118.2436 # Coordinates for Los Angeles
sd_lat, sd_lon = 32.7157, -117.1611 # Coordinates for San Diego
sj_lat, sj_lon = 37.3382, -121.8863 # Coordinates for San Jose

def load_csv( dest, print ):
    # df_train = pd.read_csv("ML flow train/housing_train.csv")
    # df_test = pd.read_csv("ML flow train/housing_test.csv")
    df = pd.read_csv(dest)
    if print == True:
        print("First rows of set:")
        print(df.head())
    
    return df

def validate_data( df ):
    # Check for missing values
    if df.isnull().sum() == 0:
        return True
    return False

def printHistplot(title, df, colName):
    sns.histplot(df[colName], kde=True)
    plt.title(title)
    plt.show()

def printBoxPlot(title, df, colName):
    sns.boxplot(x=df[colName])
    plt.title(title)
    plt.show()

def printFeatureCorrelation(title, df):
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm',fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.show()

def printScatterPlot(title, df, xColName, yColName, zColName):
    # define size
    plt.figure(figsize=(10, 6))
    # create the plot
    sns.scatterplot(
        x=xColName,
        y=yColName,
        hue=zColName,
        palette='viridis',
        data=df,
        alpha=0.6
    )

    # Add labels and title
    plt.title(title)
    plt.xlabel(xColName)
    plt.ylabel(yColName)
    plt.legend(title=zColName, loc='upper right')
    plt.show()

# Proximity between two points
def euclidean_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def add_location_clusters(df, n_clusters=10):
    coords = df[['Latitude', 'Longitude']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
    df['location_cluster'] = kmeans.predict(coords)
    df = pd.get_dummies(df, columns=['location_cluster'], drop_first=True)
    return df

def distance_featuring(df):
    # dist to big cities
    df['dist_sf'] = euclidean_distance(df['Latitude'], df['Longitude'], sf_lat, sf_lon)
    df['dist_la'] = euclidean_distance(df['Latitude'], df['Longitude'], la_lat, la_lon)
    df['dist_sd'] = euclidean_distance(df['Latitude'], df['Longitude'], sd_lat, sd_lon)
    df['dist_sj'] = euclidean_distance(df['Latitude'], df['Longitude'], sj_lat, sj_lon)

    # dist to closest city
    df['min_dist_to_city'] = df[['dist_sf', 'dist_la', 'dist_sd', 'dist_sj']].min(axis=1)
    return df

def engineer_features(df): 
    # Distance to big cities
    df = distance_featuring(df)
    # df = add_location_clusters(df)

    df['room_bedroom_rat'] =df['AveRooms'] / df['AveBedrms']
    df['income_room_interaction'] = df['MedInc'] * df['room_bedroom_rat']
    df['income_distance_interaction'] = df['MedInc'] * df['min_dist_to_city']
    
    return df

def feature_scaling(df, features_to_scale, scaler):  
    if scaler == None:
        scaler =  StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    return df

def train_model(df, features, target):
    X_train = df[features]
    y_train = df[target]

    # model = LinearRegression()
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        random_state=42 )
    model.fit(X_train, y_train)
    return model

def grid_train_model(df, features, target):
    X_train = df[features]
    y_train = df[target]

    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7]
    }

    grid_search = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    return best_model

def predict(df, features, target, model):
    X_test = df[features]
    y_test = df[target]

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

df_train = load_csv("ML flow train/housing_train.csv", False )
df_test = load_csv("ML flow train/housing_test.csv", False )

df_train = engineer_features(df_train)
df_test = engineer_features(df_test)

scaler = MinMaxScaler()
features = [
    'min_dist_to_city',
    'room_bedroom_rat',
    'MedInc',
    # 'AveRooms',
    # 'AveBedrms',
    'income_room_interaction',
    'income_distance_interaction']

df_train = feature_scaling(df_train, features, scaler)
df_test = feature_scaling(df_test, features, scaler)

features_to_train = [
    'min_dist_to_city',
    'room_bedroom_rat',
    'MedInc',
    # 'AveRooms',
    # 'AveBedrms',
    'income_room_interaction',
    'income_distance_interaction']

# model = train_model(df_train, features_to_train, 'MedHouseVal' )
model = grid_train_model(df_train, features_to_train, 'MedHouseVal' )

printFeatureCorrelation("Feature correlation",df_train )
predict(df_test, features_to_train, 'MedHouseVal', model )