## 2. Introduction to the data ##

import pandas as pd

dc_listings = pd.read_csv('dc_airbnb.csv')
print(dc_listings.head(1))

## 4. Euclidean distance ##

import numpy as np

acc = 3
acc_first_row = dc_listings['accommodates'][0]
first_distance = abs(acc - acc_first_row)


## 5. Calculate distance for all observations ##

acc = 3
dc_listings['distance'] = dc_listings['accommodates'].apply(lambda x: abs(acc-x))
dc_listings['distance'].value_counts()

## 6. Randomizing, and sorting ##

import numpy as np
np.random.seed(1)
new_index = np.random.permutation(len(dc_listings))
new_listings = dc_listings.loc[new_index]
dc_listings=new_listings.sort_values('distance')

## 7. Average price ##

stripped_commas = dc_listings['price'].str.replace(',', '').str.replace('$', '').astype(float)
dc_listings['price'] = stripped_commas
mean_price = dc_listings['price'].iloc[0:5].mean()

## 8. Function to make predictions ##

# Brought along the changes we made to the `dc_listings` Dataframe.
dc_listings = pd.read_csv('dc_airbnb.csv')
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]

def predict_price(new_listing):
    temp_df = dc_listings.copy()
    ## Complete the function.
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: abs(x-new_listing))
    temp_df=temp_df.sort_values('distance')
    top_5 = temp_df.iloc[0:5]['price']
    predicted_price = top_5.mean()
    return(predicted_price)

acc_one = predict_price(1)
acc_two = predict_price(2)
acc_four = predict_price(4)