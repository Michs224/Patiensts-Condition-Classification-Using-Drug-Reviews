# Patiensts Condition Classification Using Drug Reviews
 Patienst's Condition Classification Using Drug Reviews

---

#### Overview

This project aims to classify drug reviews based on the condition they are treating using various machine learning models. The dataset used contains drug reviews from Drugs.com, where each review is labeled with a specific medical condition.

---

#### Dataset

The dataset consists of two main files:
- `drugsComTrain_raw.tsv`: Training data containing drug reviews and associated information.
- `drugsComTest_raw.tsv`: Testing data for evaluating the performance of the models.

The dataset was preprocessed to combine both training and testing data into a single dataset and filtered to include only conditions with a significant number of reviews (>= 4000).

---

#### Features Used

1. **Bag of Words (Count Vectorizer and TF-IDF)**
   - Count Vectorizer and TF-IDF were used to convert text data (drug reviews) into numerical features.
   - Count Vectorizer counts the frequency of each word in the document.
   - TF-IDF (Term Frequency-Inverse Document Frequency) adjusts the count by how common a word is across all documents.

2. **Word Embeddings (Word2Vec)**
   - Word2Vec was employed to capture semantic meanings of words by learning dense vector representations.
   - Preprocessed text data was used to train the Word2Vec model, generating word embeddings with a dimension of 250.

---

#### Models Used

Several classification models were trained and evaluated:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Machines (SVM)**
- **Passive Aggressive Classifier**
- **Logistic Regression**
- **Random Forest**
- **XGBoost**

For each model, training was performed on the features derived from both Bag of Words (Count Vectorizer and TF-IDF) and Word2Vec embeddings.

---

#### Evaluation Metrics

The models were evaluated using the following metrics:
- **Accuracy Score**: Measures the proportion of correctly classified instances.
- **Classification Report**: Provides precision, recall, F1-score, and support for each class.
- **Confusion Matrix**: Visual representation of the model's performance showing true positives, false positives, true negatives, and false negatives.

---

#### Files Included

- `Main.ipynb`: Jupyter Notebook containing the entire project code including data preprocessing, model training, evaluation, and results visualization.
- `DrugsComPatient_raw.csv`: Combined dataset after preprocessing.
- `Models/`: Directory containing saved trained models (`xgb_tfidf_(1,2)-gram_model.pkl`, `lr_tfidf_(1,2)-gram_model.pkl`, etc.)

---

#### Dependencies

Ensure you have the following Python libraries installed to run the code:
- pandas
- numpy
- sklearn
- seaborn
- matplotlib
- xgboost
- gensim
- nltk
- BeautifulSoup

---

#### Running the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repository.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open `Main.ipynb` in Jupyter Notebook or any compatible IDE.
   
4. Run each cell sequentially to execute data preprocessing, model training, and evaluation.
   
5. For Streamlit UI, run App.py.

---
