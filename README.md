# Fake-News_Detection
Download the News Dataset from here: https://drive.google.com/file/d/1ImaByuuukwIPieCSGGRWTWSWa_lWkwUq/view?usp=sharing

This project aims to detect fake news articles using machine learning techniques. It leverages Natural Language Processing (NLP) and various classifiers to predict the authenticity of news articles as either real or fake.

## Project Overview

The project involves several key steps:

- **Data Collection and Preprocessing**:
  - Data sourced from two datasets: real news (`True.csv`) and fake news (`Fake.csv`).
  - Preprocessing steps include combining title and text, removing stopwords, and tokenizing words using NLTK and Gensim libraries.

- **Model Training**:
  - **Deep Learning Model**: Sequential model with an Embedding layer, Bidirectional LSTM layer, and Dense layers for classification.
  - **Traditional ML Classifiers**: Decision Tree, Passive Aggressive, Random Forest, Logistic Regression, Naive Bayes, and k-Nearest Neighbors (kNN) classifiers.
  - Evaluation metrics used include accuracy scores and confusion matrices.

- **Comparison of Classifiers**:
  - Results showed high accuracy across all models, with LSTM achieving the highest accuracy of 99.82%.

- **Deployment**:
  - The trained LSTM model (`mymodel.h5`) is available for deployment to detect fake news in real-time.

- **Usage**:
  - The project offers a function `fake_news_det(news)` for predicting the authenticity of a news article input.

- **Dependencies**:
  - Python libraries used include NLTK, Gensim, TensorFlow, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.
  - Ensure dependencies are installed using `requirements.txt` or individually installed via `pip`.

- **File Structure**:
  - `main.ipynb`: Jupyter Notebook containing the complete code and analysis.
  - `data/True.csv` and `data/Fake.csv`: Datasets used for training and testing.
  - `mymodel.h5`: Saved LSTM model for deployment.

- **Further Improvements**:
  - Explore ensemble techniques to further improve model accuracy.
  - Implement advanced NLP techniques for feature extraction and sentiment analysis.

## License

This project is licensed under the MIT License. Feel free to modify and distribute it as needed.

## Acknowledgments

- Special thanks to NLTK, Gensim, TensorFlow, and Scikit-learn communities for their open-source contributions.
- Dataset sources: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

For any inquiries or support, please contact Abdul Samad.
