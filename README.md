# Big Data Analysis Project: Wikipedia Articles Classification

## Overview

This project analyzes a large dataset of Wikipedia articles to classify them into different categories and extract meaningful insights. The analysis pipeline includes exploratory data analysis, text preprocessing, feature extraction, and the development of a classification model using both traditional and Spark-based machine learning techniques.

## Features

- **Data Acquisition**: Downloads and processes a Wikipedia dataset containing 153,232 articles
- **Exploratory Data Analysis**: Examines article distribution across categories and basic statistics
- **Text Visualization**: Creates word clouds and frequency histograms for each category
- **Scalable Processing**: Implements data preprocessing and model training using Apache Spark
- **Multi-Feature Classification**: Uses both article summaries and full content for improved accuracy
- **Performance Metrics**: Calculates and reports detailed classification metrics by category
- **Content Analysis**: Analyzes linguistic patterns and frequent words across different categories

## Technologies Used

- **Python**: Core programming language
- **PySpark**: For distributed data processing and ML pipelines
- **Pandas/NumPy**: For data manipulation and analysis
- **NLTK**: For natural language processing
- **Matplotlib/Seaborn**: For data visualization
- **WordCloud**: For text visualization
- **Scikit-learn**: For ML algorithm implementation

## Project Structure

The project is structured in several main components:

1. **Setup and Data Loading**: 
   - Installation of dependencies
   - Download of the Wikipedia dataset
   - Initial data inspection

2. **Exploratory Data Analysis**:
   - Basic text cleaning and preprocessing
   - Category distribution analysis
   - Word frequency analysis
   - Statistical analysis by category

3. **Data Preparation with PySpark**:
   - Conversion of Pandas DataFrame to Spark DataFrame
   - Text tokenization and feature extraction
   - TF-IDF transformation
   - Training and test set splitting

4. **Classification Model Development**:
   - Feature engineering for both summary and full text
   - Vector assembly and model training
   - Logistic regression implementation
   - Model evaluation and metrics calculation

5. **Content Analysis**:
   - Article density by category
   - Linguistic trend analysis
   - Frequent word identification by category

## Results

The classification model achieved excellent results:
- Overall accuracy exceeding 91%
- Most categories classified with F1-Scores above 90%
- Only "medicine" and "research" categories showing slightly lower performance (around 75% F1-Score)

The content analysis revealed:
- Nearly uniform distribution of articles across categories, with a slight predominance of "politics"
- Strong linguistic patterns characterizing each category:
  - "power" and "plant" appear in over 60% of energy articles
  - "tennis" dominates the sport category
  - "university" and "research" are prevalent in the research category (over 80% of articles)

## How to Run

1. **Environment Setup**:
   ```bash
   pip install wordcloud nltk pandas matplotlib seaborn
   pip install pyspark scikit-learn
   ```

2. **Dataset Acquisition**:
   The dataset will be automatically downloaded from:
   ```
   https://proai-datasets.s3.eu-west-3.amazonaws.com/wikipedia.csv
   ```

3. **Run the Analysis**:
   Execute the notebook cells in order to perform:
   - Exploratory data analysis
   - Model training and evaluation
   - Content analysis

4. **Prediction on New Articles**:
   Use the `predici_categoria` function by passing:
   ```python
   categoria_predetta = predici_categoria(model, "article_text", "article_summary")
   ```

## Future Work

Potential improvements and extensions:
- Implement more sophisticated NLP techniques (lemmatization, entity recognition)
- Explore deep learning approaches (BERT, transformers)
- Add topic modeling to discover hidden themes in each category
- Create an interactive dashboard for result visualization
- Develop an API for real-time article classification

## Requirements

- Python 3.6+
- Apache Spark 3.0+
- Libraries: pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, wordcloud
- 2+ GB of RAM recommended
