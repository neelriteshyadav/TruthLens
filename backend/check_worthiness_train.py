import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Example dataset
data = pd.read_csv('./data/consolidated.csv')
# Split data into features and target
X = data['Text']  # Text data (feature)
y = data['Verdict']  # Labels (target)

def predict_claim(query):
    # Split into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = CountVectorizer()

    # Fit and transform the training data (to learn the vocabulary and vectorize)
    X_train_vect = vectorizer.fit_transform(X_train)

    # Transform the test data (use the learned vocabulary from the training data)
    X_test_vect = vectorizer.transform(X_test)

    # Initialize the classifier
    classifier = LogisticRegression()

    # Train the classifier on the training data
    classifier.fit(X_train_vect, y_train)

    # Make predictions on the test data
    y_pred = classifier.predict(X_test_vect)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    new_query = [query]

    # Transform the new query using the same vectorizer
    new_query_vect = vectorizer.transform(new_query)


    # Predict the labels (0 or 1)
    predictions = classifier.predict(new_query_vect)

    return predictions[0]



