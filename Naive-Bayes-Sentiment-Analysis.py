from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the text file
file_path = input("Enter the path to the text file: ")
with open(file_path, 'r') as file:
    text = file.read()

# Create a DataFrame with the text
data = {'text': [text]}
input_data = pd.DataFrame(data)

 
# Load the data
full_data = pd.read_csv("data/movie.csv")

# Split the data into training and test sets
trainData, testData = train_test_split(full_data, test_size=0.2, random_state=42)

# Create feature vectors
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
train_vectors = vectorizer.fit_transform(trainData['text'])
test_vectors = vectorizer.transform(testData['text'])
input_vectors = vectorizer.transform(input_data['text'])
# Perform classification with Naive Bayes
classifier_nb = MultinomialNB()
classifier_nb.fit(train_vectors, trainData['label'])
prediction_nb = classifier_nb.predict(input_vectors)

if prediction_nb[0] == 1:
    print("Sentiment: Positive")
elif prediction_nb[0] == 0:
    print("Sentiment: Negative")

# Generate classification report for test data
test_predictions = classifier_nb.predict(test_vectors)
print("\nClassification Report for Test Data:")
print(classification_report(testData['label'], test_predictions))
