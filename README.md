# 🎬 Movie Success Predictor

A machine learning web application that predicts movie box office success using a Random Forest model trained on 540+ movies.

## Deployed Link

[https://movie-success-predictor.up.railway.app](https://movie-success-predictor.up.railway.app)

## Features

- **🎯 Movie Success Prediction**: Interactive form to predict if a movie will be successful
- **📊 Data Insights**: Comprehensive analysis dashboard with visualizations
- **🧠 Machine Learning**: 81.5% accuracy with Random Forest model
- **📈 Statistical Analysis**: Chi-square tests and sentiment analysis
- **🎨 Modern UI**: Beautiful, responsive web interface

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd movie-analysis
   ```

2. **Create and activate the virtual environment**
   ```bash
   python -m venv movie-analysis-env
   movie-analysis-env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Make sure you're in the virtual environment**
   ```bash
   movie-analysis-env\Scripts\activate
   ```

2. **Run the application**
   ```bash
   python app.py
   ```

3. **Open your browser** and go to `http://localhost:5000`

## Running the Jupyter Notebook

1. **Open the notebook**
   - Open `movie-analysis.ipynb` in your directory
   - Select the `movie-analysis-env` kernel from the kernel selector
   - Click the **Run All** button at the top to execute all cells

## How It Works

1. **Data Cleaning**: 
   - Fixed runtime zeros with mean imputation
   - Corrected budget unit inconsistencies
   - Used hot-deck imputation for star count errors

2. **Statistical Testing**:
   - Performed chi-square tests to identify significant relationships

3. **Feature Engineering**:
   - Sentiment analysis on movie reviews using TextBlob
   - One-hot encoding for categorical variables
   - 20+ features including budget, stars, runtime, season, genre, etc.

4. **Machine Learning**:
   - Random Forest model with 81.5% accuracy
   - Trained on 90% of data, tested on 10%

5. **Web Interface**:
   - Flask backend serves the ML model via REST API
   - Modern HTML, Bootstrap/CSS, JavaScript frontend
   - Real-time predictions and interactive visualizations

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Bootstrap 5
- **Data Cleaning**: Pandas, Numpy
- **Statistical Testing**: SciPy
- **Data Visualization**: Seaborn
- **ML Libraries**: Scikit-Learn
- **Sentiment Analysis**: TextBlob
- **Deployed**: Railway
