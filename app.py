import streamlit as st
import numpy as np
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

def main():

    st.title('car_prediction')
    st.sidebar.title('Car_prediciton')



    def load_data():
        data=pd.read_csv('car data.csv')
        return data


    def processing_data(df):
        df.drop(['Car_Name'],axis=1,inplace=True)
        df['no_of_yr']=2020-df['Year']
        df.drop('Year',axis=1,inplace=True)
        df=pd.get_dummies(df,drop_first=True)
        return df
    def model(p_df):
        X=p_df.drop(['Selling_Price'],axis=1)
        y=p_df['Selling_Price']
        model = ExtraTreesRegressor()
        model.fit(X,y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        regressor=RandomForestRegressor()
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
        min_samples_split = [2, 5, 10, 15, 100]
        min_samples_leaf = [1, 2, 5, 10]

        random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
        rf = RandomForestRegressor()
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
        rf_random.fit(X_train,y_train)
        predictions=rf_random.predict(X_test)
        return predictions,X_train, X_test, y_train, y_test,rf_random
        file = open('random_forest_regression_model.pkl', 'wb')
        pickle.dump(rf_random, file)

    df=load_data()
    p_df=processing_data(df)
    if st.sidebar.checkbox('show processed data'):
        st.subheader('car_data')
        st.write(p_df)
    if st.sidebar.checkbox('show corr heatmap'):
        st.subheader('corr_heatmap')
        st.write(sns.heatmap(p_df.corr(),annot=True,cmap="RdYlGn"))
        st.pyplot()
    if st.sidebar.checkbox('scatter plot'):
        st.subheader('scatter plot')
        st.write(plt.scatter(y_test,pred))
        st.pyplot()
    Year=st.sidebar.number_input('total yr of purchase',0,50,value=0,step=1)
    present_price=st.sidebar.number_input('what is the showroom price in lakhs',0.0,50.0,step=1.0)
    km_driven=st.sidebar.number_input('how many km driven?',0,50000,value=0,step=1000)
    owner=st.sidebar.number_input('how many owners previously had the car 0,1 or 3 ?',0,3,value=0,step=1)
    fuel=st.sidebar.selectbox('what is the fuel type?',('petrol','diesel'))
    type=st.sidebar.selectbox('are yu a dealer or individual?',('dealer','individual'))
    transmission_type=st.sidebar.selectbox('transmission type',('manual','auto'))

    if st.sidebar.button('selling_price'):
        Year=Year
        Present_Price=present_price
        Kms_Driven= km_driven
        Owner=owner
        if fuel=='petrol':
            Fuel_Type_Diesel=0
            Fuel_Type_Petrol=1
        else:
            Fuel_Type_Diesel=1
            Fuel_Type_Petrol=0
        if type=='dealer':
            Seller_Type_Individual=0
        else:
            Seller_Type_Individual=1
        if transmission_type=='manual':
            Transmission_Manual=1
        else:
            Transmission_Manual=0

        pred,X_train, X_test, y_train, y_test,rf_random=model(p_df)
        dfc=[Present_Price,Kms_Driven,Owner,Year,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Manual]
        predicted=rf_random.predict([dfc])
        st.write('selling price for the car is',predicted)




if __name__ =='__main__':
    main()
