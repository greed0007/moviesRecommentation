# moviesRecommentation

![image](https://github.com/gerhea/moviesRecommentation/assets/73679634/319b4f51-7191-4ac8-999c-e00697781600)


***Data Source:***

Data Source URL: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts, and vote averages. This dataset also has files containing 26 million ratings from 270,000 users for all 45,000 movies. Ratings are on a scale of 1-5 and have been obtained from the official GroupLens website.

***Project Description***

My project aims to create a movie recommendation system using machine learning techniques and Apache Spark. I will be utilizing the Full MovieLens dataset from Kaggle, which includes metadata for 45,00 movies and 26 million ratings from 270,000 users. The recommendation system will be able to suggest movies to users based on their past ratings and viewing history.

To achieve this, I will first preprocess the data by cleaning and merging the various files in the dataset. Use collaborative filtering techniques such as Alternating Least Squares to train a model on the rating data. This model will be able to predict how a user would rate a movie they have not seen yet, based on similar preferences

I will learn the necessary techniques and tools through hands-on experimentation in Google Colab and reference documentation from Apache Spark.

The end goal of this project is to create a functional movie recommendation system that suggests movies to a user based on their individual preferences and viewing history.

https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html


***Project Abstract***

This project aims to build a movie recommender system using collaborative filtering techniques to predict user ratings and make personalized recommendations. The dataset used is the MovieLens dataset, which contains information on over 27,000 movies and 138,000 ratings. The collaborative filtering algorithm is implemented using the Apache Spark MLlib library and evaluated using root-mean-square error (RMSE) and Precision-Recall (PR). The final model achieves an RMSE of 0.8475 and a Precision-Recall PR value of 0.9968. Overall, this project demonstrates the effectiveness of collaborative filtering in building personalized recommender systems for movie ratings.

***Project Milestone I***
The progress I was able to accomplish so far was several tasks using PySpark, using PySpark SQL and creating dataframes for movies metadata, keywords, credits, and ratings. The ratings data has also been cleaned by dropping rows with null values in the userid and movie columns. The movies metadata, ratings, and credits dataframes have been joined into a single dataframe, which has been split into training and test sets. An ALS model has been built and fitted to training data, and test sets referencing ALS Apache Spark Documentation. The model has been evaluated using Root MEan Squared Error (RMSE), and the distribution of movie ratings has been visualized using histograms and boxplots.

Then next step is to use the trained ALS model to recomment movies to users based on their ratings history by predicting the ratings that the usr would give to a particular movie and recommending movies with the highest predicted ratings.

Plan for rest of semester:

Get a list of movies that user has not rated by either filtering movie_ratings dataframe to include only movies that the user has not rated.
Use the ALS model to predict the ratings that the user would give to each of the unrated movies by calling the transform() method on the ALS model and passing in a dataframe that contains the user ID and the movie Ids of the unrated movies.
Sort the predicted ratings in descending order and select the top N movies with the highest predicted ratings as recommendations for the user.
https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.recommendation.ALS.html

https://spark.apache.org/docs/3.2.0/api/python/reference/api/pyspark.sql.functions.transform.html

Project Milestone II
I was unable to attempt what I planned in milestone one because this dataset did not have movies rating that equals zero. The progress I was able to accomplish so far was several tasks using PySpark, matplotlib, and pandas for some visualization. The objective of the project was to analyze movie ratings data, filter movies by release year, and make recommendations using the ALS model. In doing that I was able to analzye the movie ratings data that was loaded and cleaned to remove duplicates and missing values. Create a histogram and boxplot of the movie ratings were plotted to visualize the distribution and outliers. I experimented with filitering by release year to focus on recent movies. Recommendations were made for all users using the ALS model. The recommended movies were joined with the movies_metadata dataset to get the titles and release years. The recommendations were further filtered to only include movies with a minimum rating of 4. Personalized recommendations were made for a specific user by selecting movies the user hasn't seen yet using the ALS model. I then created a scatter plot of the number of recommended movies by their release year. In addidtion to chartting the average popularity of movies by genre.

Plan for the rest of semester:

Try to implement something for error handing.
