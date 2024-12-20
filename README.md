# Twitter Sentiment Analysis

## Overview
This project performs sentiment analysis on tweets using the **Multinomial Naive Bayes** algorithm. The goal is to classify tweets into **Positive**, **Negative**, **Neutral**, and **Irrelevant** sentiments. It involves preprocessing tweet data, training a Naive Bayes model, and evaluating its performance using various metrics.

## Table of Contents
1. [Project Setup](#project-setup)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [Results & Visualizations](#results--visualizations)
6. [Future Improvements](#future-improvements)

## Project Setup
### Prerequisites
- Python 3.x
- Libraries:
  - `pandas`
  - `re`
  - `matplotlib`
  - `seaborn`
  - `textblob`
  - `wordcloud`
  - `sklearn`
  - `nltk`

### Installation
You can install the necessary libraries using `pip`:
```bash
pip install pandas matplotlib seaborn textblob wordcloud scikit-learn nltk
```

### Dataset
The datasets used in this project are:
- **Twitter Training Dataset**: Contains tweet texts and their respective sentiment labels.
- **Twitter Validation Dataset**: Used for model validation.

### File Structure
```
.
├── twitter_training.csv
├── twitter_validation.csv
├── Twitter_Sentiment_Analysis.ipynb
└── README.md
```

## Data Preprocessing
The dataset is preprocessed in the following steps:
1. **Cleaning the tweets**: URLs, mentions, hashtags, and special characters are removed.
2. **Tokenization**: The text is split into individual tokens (words).
3. **Lowercasing**: All text is converted to lowercase.
4. **Stopword Removal**: Common English stopwords are removed using NLTK.

```python
# Function to clean tweets
def clean_tweet(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtag symbol
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text
```

## Model Training
The **Multinomial Naive Bayes** algorithm is used to classify the sentiment of tweets:
1. The cleaned text is vectorized using **CountVectorizer** to convert words into numerical features.
2. The data is split into training and testing sets using **train_test_split** from Scikit-learn.
3. The model is trained on the training data, and predictions are made on the test data.

```python
# Initialize and train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)
```

## Model Evaluation
Model performance is evaluated using:
1. **Classification Report**: Includes precision, recall, and F1-score for each sentiment class.
2. **Confusion Matrix**: Shows the model’s ability to correctly classify each sentiment.
3. **Cross-Validation**: To assess if the model is overfitting, 5-fold cross-validation is used.

```python
# Cross-validation to check for overfitting
cv_scores = cross_val_score(model, X, y, cv=5)
```

## Results & Visualizations
### 1. **Sentiment Distribution**
A **bar plot** is created to visualize the distribution of sentiments across the training dataset.

```python
# Count the occurrences of each sentiment label
sentiment_counts = df_train['sentiment'].value_counts()

# Visualize the result
plt.figure(figsize=(10,7))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()
```

### 2. **Word Cloud**
A **word cloud** is generated for each sentiment to highlight the most common words in tweets of that sentiment.

```python
# Function to generate word cloud
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=15)
    plt.axis('off')
    plt.show()

# Generate word clouds for each sentiment
for sentiment in df_train['sentiment'].unique():
    text = ' '.join(df_train[df_train['sentiment'] == sentiment]['cleaned_text'])
    generate_wordcloud(text, f'Word Cloud for {sentiment.capitalize()} Tweets')
```

### 3. **Confusion Matrix**
The **confusion matrix** is visualized using a heatmap to understand the model's performance in classifying tweets into the correct sentiment category.

```python
# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred, labels=['Positive', 'Negative', 'Neutral', 'Irrelevant'])

# Visualize the result
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=['Positive', 'Negative', 'Neutral', 'Irrelevant'], yticklabels=['Positive', 'Negative', 'Neutral', 'Irrelevant'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')
plt.show()
```

## Future Improvements
- Experiment with other **classification algorithms** like **Logistic Regression** or **SVM** to potentially improve performance.
- Perform **hyperparameter tuning** to optimize the Naive Bayes model.
- Implement **deep learning** models, like LSTM or BERT, for improved sentiment classification.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- The datasets used in this project were sourced from Kaggle.
- The **NLTK** library was used for stopwords removal.
