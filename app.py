from flask import Flask, render_template , request
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error,mean_absolute_error

# Read the data from a CSV file
df = pd.read_csv('df.csv')


# Preprocess the data
df['Location'] = df['Location'].str.lower()
df['Cuisine'] = df['Cuisine'].str.lower()


# Create a TF-IDF vectorizer for cuisine features
cuisine_vectorizer = TfidfVectorizer(stop_words='english')
cuisine_features = cuisine_vectorizer.fit_transform(df['Cuisine'])

# Create the Flask application
app = Flask(__name__)

# Scale the price values
price_scaler = StandardScaler()
df['Price_4_two'] = price_scaler.fit_transform(df['Price_4_two'].values.reshape(-1, 1))

# Define the routes

@app.route('/')
def index():
    return render_template('index.html',df=df)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html',df=df)

@app.route('/reco' , methods=['POST'])
def reco():
    user_location = request.form.get('user_location')
    user_cuisine = request.form.get('user_cusine')
    user_price = request.form.get('user_price')


    user_price = float(user_price)
    user_location=user_location.lower()
    user_cuisine=user_cuisine.lower()

    # Create a TF-IDF vectorizer for cuisine features
    cuisine_vectorizer = TfidfVectorizer(stop_words='english')
    cuisine_features = cuisine_vectorizer.fit_transform(df['Cuisine'])

    # Find average price for one person in the given location
    average_price = df[df['Location'] == user_location]['price_for_two'].mean()
    average_price=round(average_price,0)

    # Find popular cuisine in the given location
    popular_cuisine = df[df['Location'] == user_location]['Cuisine'].value_counts().index[0]

    # Find the most popular restaurant in the given location
    most_popular_restaurant = df[df['Location'] == user_location]['Restaurant'].value_counts().index[0]

    # Find the most popular restaurant with the user's preferred cuisine and location
    result = df[(df['Location'] == user_location) & (df['Cuisine'].str.contains(user_cuisine, case=False))]

    if result.empty:
        most_popular_cuisine_restaurant = f"Sorry, we could not find any restaurants in {user_location} serving {user_cuisine}."
    else:
        most_popular_cuisine_restaurant = result['Restaurant'].value_counts().index[0]

    # Suggest a price
    s_average_price = df[df['Location'] == user_location]['price_for_two'].mean()

    if user_price < s_average_price:
        suggested_price = round(user_price * 1.1)
    else:
        suggested_price = round(s_average_price)

    # Find similar restaurants
    similar_restaurants = []
    for idx in range(cuisine_features.shape[0]):
        restaurant = df.iloc[idx]
        if user_cuisine.lower() in restaurant['Cuisine'].lower():
            score = cosine_similarity(cuisine_features[idx], cuisine_vectorizer.transform([user_cuisine]))[0][0] + \
                    price_scaler.transform([[restaurant['Price_4_two']]])[0][0]
            similar_restaurants.append((restaurant['Restaurant'], score))

    # Sort the restaurants by similarity score

    Rec_Restaurants = list(set([f'{restaurant} (Score: {score:.2f})' for restaurant, score in similar_restaurants]))
    sorted_restaurants = sorted(Rec_Restaurants, key=lambda x: float(x.split('(Score: ')[1][:-1]), reverse=True)

    Rec_Restaurants = sorted_restaurants[:5]

    result = df[(df['Location'] == user_location) & (df['Cuisine'].str.lower() == user_cuisine) & (
                df['Price_4_two'] <= user_price)]

    #new way
    if result.empty:
        cf_rec_restaurants = nmf_rec_restaurants =(f"Sorry, no restaurants found matching your preferences.")
    else:
        # Matrix Factorization Recommendation
        result['Price_4_two'] = result['Price_4_two'].abs()
        nmf_model = NMF(n_components=100)
        nmf_features = nmf_model.fit_transform(result[['Price_4_two']])
        nmf_similarities = nmf_features.dot(nmf_features.T)

        # Get the top 5 most similar restaurants
        nmf_indices = np.argsort(nmf_similarities[-1])[::-1][1:6]
        nmf_rec_restaurants = result.iloc[nmf_indices]['Restaurant'].tolist()
        nmf_rec_restaurants =sorted(nmf_rec_restaurants)
        nmf_accuracy = np.mean(nmf_similarities[nmf_indices, -1])


        # Calculate MSE and MAE for Matrix Factorization

        test_X = result[['Price_4_two']].values
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        test_pred = nmf_model.inverse_transform(nmf_model.transform(test_X))
        nmf_mse = mean_squared_error(test_X, test_pred)
        nmf_mae = mean_absolute_error(test_X, test_pred)

        # Collaborative Filtering Recommendation
        cuisine_vectorizer = TfidfVectorizer()
        cuisine_features = cuisine_vectorizer.fit_transform(result['Cuisine'])
        cosine_similarities = cosine_similarity(cuisine_features)

        # Get the top 5 most similar restaurants
        cf_indices = np.argsort(cosine_similarities[-1])[::-1][1:6]
        cf_rec_restaurants = result.iloc[cf_indices]['Restaurant'].tolist()
        cf_rec_restaurants = sorted(cf_rec_restaurants)
        cf_accuracy = np.mean(cosine_similarities[cf_indices, -1])


        from sklearn.metrics import mean_squared_error, mean_absolute_error
        # Calculate MSE and MAE for Collaborative Filtering
        test_X = cuisine_features.toarray()[-1].reshape(1, -1)
        test_pred = cosine_similarity(cuisine_features.toarray()[:-1], test_X)
        cf_mse = mean_squared_error(cosine_similarities[:-1, -1], test_pred)
        cf_mae = mean_absolute_error(cosine_similarities[:-1, -1], test_pred)





    print(user_price)
    print(user_location)
    print(user_cuisine)
    print(cf_mse)
    print(cf_mae)
    print(nmf_mse)
    print(nmf_mae)
    print(nmf_accuracy)
    print(cf_accuracy)
    print(cf_rec_restaurants)
    print(nmf_rec_restaurants)
    print(Rec_Restaurants)
    print(suggested_price)
    print(average_price)
    print(most_popular_restaurant)
    print(popular_cuisine)
    print(most_popular_cuisine_restaurant)
    return render_template('recommend.html',user_cuisine=user_cuisine,user_location=user_location, user_price=user_price, nmf_mse=nmf_mse ,nmf_mae=nmf_mae ,cf_mae=cf_mae ,cf_mse=cf_mse , cf_accuracy=cf_accuracy ,nmf_accuracy=nmf_accuracy ,cf_rec_restaurants=cf_rec_restaurants ,nmf_rec_restaurants=nmf_rec_restaurants , df=df , Rec_Restaurants=Rec_Restaurants, suggested_price=suggested_price , average_price=average_price , popular_cuisine=popular_cuisine , most_popular_restaurant=most_popular_restaurant , most_popular_cuisine_restaurant=most_popular_cuisine_restaurant)

if __name__ == '__main__':
    app.run(debug=True)
