
# Sentiment Analysis Model

In this project I used the `nltk` library for natural language processing to create a model that classifies any inputs from 
keyboard into `Positive` or `Negative` based on the attitude of the writer to the topic. This model is trained using the 
`nltk.twitter_samples` dataset with the Naive Bayes Classifier.

To generate the model:

```Makefile
make generate
```

To run the main file:
```Makefile
make run
```