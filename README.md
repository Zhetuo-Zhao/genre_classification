# Genre classification
We compared different methods, including KNN, SVM, DNN, RNN, LSTM, to classify music genres in GTZAN genre collection dataset. 

We used features in dimension of 57 of every 3 seconds as the input. Therefore, for each song of 30 seconds, we have 57x10 features. For those features, we can treat them as independent features and using KNN, SVM, DNN method to build classifiers. Alternative, we can also use recurrent models like RNN and LSTM to dig out how those features changes in time in a song. 
