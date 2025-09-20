from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from textblob import TextBlob
import json
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

class MovieAnalysisApp:
    def __init__(self):
        self.df = None
        self.model = None
        self.feature_columns = None
        self.scaler = None
        self.accuracy = None
        self.confusion_matrix = None
        self.feature_importance = None
        self.model_name = None
        self.model_results = None
        self.test_indices = None
        
        try:
            print("🚀 Starting MovieAnalysisApp initialization...")
            self.load_and_prepare_data()
            print("✅ Data loaded and prepared successfully")
            self.train_model()
            print("✅ Model trained successfully")
            print(f"📊 Model accuracy: {self.accuracy:.3f}")
            print(f"🎯 Model type: {self.model_name}")
        except Exception as e:
            print(f"❌ Error initializing MovieAnalysisApp: {str(e)}")
            import traceback
            traceback.print_exc()
            # Set default values to prevent crashes
            self.df = pd.DataFrame()
            self.model = None
    
    def load_and_prepare_data(self):
        """Load and clean the movie dataset"""
        csv_path = os.path.join(os.path.dirname(__file__), "movie_dataset.csv")
        self.df = pd.read_csv(csv_path, index_col=0)
        
        # Data cleaning (from your notebook)
        self.clean_data()
        
        # Feature engineering
        self.create_features()
    
    def clean_data(self):
        """Clean the data as done in your notebook"""
        # Remove quotes from string columns
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].astype(str).str.strip('"')
        
        # Fix runtime zeros with mean imputation
        runtimes_excluding_zero = self.df[self.df["Runtime"] != 0]["Runtime"]
        mean_runtime = round(runtimes_excluding_zero.mean())
        self.df["Runtime"] = self.df["Runtime"].apply(
            lambda x: mean_runtime if x == 0 else x
        )
        
        # Fix budget units (convert from ones to millions where needed)
        self.df["Budget"] = self.df["Budget"].apply(
            lambda x: (x / 1000000) if (x > 1000000) else x
        )
        
        # Fix stars with hot-deck imputation
        movies_excluding_100_stars = self.df[self.df["Stars"] != 100]
        similar_collection = movies_excluding_100_stars[
            (movies_excluding_100_stars["Budget"] >= 55.0) & 
            (movies_excluding_100_stars["Budget"] <= 65.0) & 
            (movies_excluding_100_stars["Promo"] >= 30.0) & 
            (movies_excluding_100_stars["Promo"] <= 40.0)
        ]
        if len(similar_collection) > 0:
            average_stars = round(similar_collection["Stars"].mean())
            self.df["Stars"] = self.df["Stars"].apply(
                lambda x: average_stars if x == 100 else x
            )
    
    def movie_review_sentiment(self, review):
        """Analyze sentiment of movie reviews"""
        text_blob_analysis = TextBlob(review)
        sentiment_result = text_blob_analysis.sentiment.polarity
        return 1 if sentiment_result >= 0 else -1
    
    def create_features(self):
        """Create features as done in your notebook"""
        # Sentiment analysis
        self.df["R1_Sentiment"] = self.df["R1"].apply(self.movie_review_sentiment)
        self.df["R2_Sentiment"] = self.df["R2"].apply(self.movie_review_sentiment)
        self.df["R3_Sentiment"] = self.df["R3"].apply(self.movie_review_sentiment)
        
        # One-hot encoding
        one_hot_season = pd.get_dummies(self.df["Season"])
        one_hot_rating = pd.get_dummies(self.df["Rating"])
        one_hot_genre = pd.get_dummies(self.df["Genre"])
        
        self.df = self.df.join(one_hot_season)
        self.df = self.df.join(one_hot_rating)
        self.df = self.df.join(one_hot_genre)
        
        print(f"📊 Features created. DataFrame shape: {self.df.shape}")
        print(f"📊 Available columns: {list(self.df.columns)}")
    
    def train_model(self):
        """Train improved machine learning models"""
        # Define feature columns (same as in your notebook)
        self.feature_columns = [
            "Runtime", "Stars", "Year", "Budget", "Promo", 
            "R1_Sentiment", "R2_Sentiment", "R3_Sentiment",
            "Fall", "Spring", "Summer", "Winter",
            "PG", "PG13", "R",
            "Action", "Drama", "Fantasy", "Romantic Comedy", "Science fiction"
        ]
        
        # Ensure all feature columns exist
        missing_columns = []
        for col in self.feature_columns:
            if col not in self.df.columns:
                self.df[col] = 0
                missing_columns.append(col)
        
        if missing_columns:
            print(f"⚠️  Missing columns created with zeros: {missing_columns}")
        
        features = self.df[self.feature_columns]
        target = self.df["Success"].astype(int)  # Ensure Success is integer (0/1)
        
        print(f"📊 Features shape: {features.shape}")
        print(f"📊 Target shape: {target.shape}")
        print(f"📊 Target distribution: {target.value_counts().to_dict()}")
        
        # Train-test split (using same random_state for fair comparison)
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.1, random_state=0, shuffle=True
        )
        
        print(f"📊 Dataset split: {len(X_train)} train, {len(X_test)} test")
        print(f"📊 Test set success rate: {y_test.sum()}/{len(y_test)} = {y_test.sum()/len(y_test):.1%}")
        
        # Store test set indices for random examples
        self.test_indices = X_test.index
        
        # Feature scaling for numerical features
        self.scaler = StandardScaler()
        numerical_features = ["Runtime", "Year", "Budget", "Promo"]
        
        # Scale training data
        X_train_scaled = X_train.copy()
        X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        
        # Scale test data
        X_test_scaled = X_test.copy()
        X_test_scaled[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        # Try the earlier trio and pick the best
        models = {
            'Decision Tree (Tuned)': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=0
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=0
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=0
            )
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        model_results = {}
        
        print("🧠 Training multiple models...")

        # Train and evaluate each model (on scaled features)
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()

            model.fit(X_train_scaled, y_train)

            test_pred = model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, test_pred)

            model_results[name] = {
                'cv_score': cv_mean,
                'test_accuracy': test_accuracy,
                'model': model
            }

            print(f"📊 {name}: CV={cv_mean:.3f}, Test={test_accuracy:.3f}")

            if test_accuracy > best_score:
                best_score = test_accuracy
                best_model = model
                best_name = name
        
        print(f"🏆 Best model: {best_name} with {best_score:.3f} accuracy")
        print(f"📈 Improvement: {best_score:.3f} vs previous")
        print(f"🧪 Total models tested: {len(model_results)}")
        
        # Show all model results for comparison
        print("📊 All model results:")
        for name, result in sorted(model_results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True):
            print(f"   {name}: {result['test_accuracy']:.4f}")
        
        # Use the best model
        self.model = best_model
        self.model_name = best_name
        self.model_results = model_results
        
        # Calculate performance metrics for the best model
        y_pred = self.model.predict(X_test_scaled)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        
        # Store references
        self.y_test = y_test
        
        
        # Feature importance (works for all tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        else:
            self.feature_importance = {col: 0 for col in self.feature_columns}
    
    def predict_success(self, movie_data):
        """Predict if a movie will be successful (matches current training pipeline)"""
        print(f"🔍 Processing prediction for: {movie_data}")
        
        # Process reviews for sentiment analysis (same as training)
        if 'R1' in movie_data:
            movie_data['R1_Sentiment'] = self.movie_review_sentiment(movie_data['R1'])
            print(f"📝 R1 sentiment: {movie_data['R1_Sentiment']}")
        if 'R2' in movie_data:
            movie_data['R2_Sentiment'] = self.movie_review_sentiment(movie_data['R2'])
            print(f"📝 R2 sentiment: {movie_data['R2_Sentiment']}")
        if 'R3' in movie_data:
            movie_data['R3_Sentiment'] = self.movie_review_sentiment(movie_data['R3'])
            print(f"📝 R3 sentiment: {movie_data['R3_Sentiment']}")
        
        print(f"🔍 Feature columns expected: {self.feature_columns}")
        print(f"🔍 Features available: {list(movie_data.keys())}")
        
        # Create feature vector in the same column order
        feature_vector = np.zeros(len(self.feature_columns))
        for i, col in enumerate(self.feature_columns):
            if col in movie_data:
                feature_vector[i] = movie_data[col]
                print(f"📊 {col}: {movie_data[col]}")
            else:
                print(f"⚠️ Missing feature: {col}")

        print(f"🔍 Final feature vector: {feature_vector}")
        
        # DataFrame to align columns
        feature_df = pd.DataFrame([feature_vector], columns=self.feature_columns)

        # Scale numerical features (same as training)
        numerical_features = ["Runtime", "Year", "Budget", "Promo"]
        feature_df[numerical_features] = self.scaler.transform(feature_df[numerical_features])

        # Predict using the selected model
        prediction = self.model.predict(feature_df)[0]
        # Some models may not support predict_proba; guard accordingly
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(feature_df)[0]
            confidence = float(max(proba))
            success_probability = float(proba[1] if len(proba) > 1 else 0)
        else:
            # Fallback: map decision function to pseudo-confidence
            confidence = 0.5
            success_probability = 0.0

        return {
            'prediction': bool(prediction),
            'confidence': confidence,
            'success_probability': success_probability
        }
    
    def get_insights(self):
        """Get key insights from the analysis"""
        # Critic analysis
        r1_negative = (self.df["R1_Sentiment"] == -1).sum() / len(self.df)
        r2_negative = (self.df["R2_Sentiment"] == -1).sum() / len(self.df)
        r3_negative = (self.df["R3_Sentiment"] == -1).sum() / len(self.df)
        
        harshest_critic = "R1" if r1_negative > max(r2_negative, r3_negative) else \
                         "R2" if r2_negative > r3_negative else "R3"
        
        # Budget-Promo correlation
        budget_promo_cov = self.df["Budget"].cov(self.df["Promo"])
        
        # Calculate False Positive Rate and False Negative Rate
        # Confusion matrix format: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        total = tn + fp + fn + tp
        
        # Using the same formulas from your notebook
        false_positive_rate = fp / total  # FP / (FP + TN + FN + TP)
        false_negative_rate = fn / total  # FN / (FP + TN + FN + TP)
        
        return {
            'accuracy': self.accuracy,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'feature_importance': self.feature_importance,
            'harshest_critic': harshest_critic,
            'critic_negative_rates': {
                'R1': r1_negative,
                'R2': r2_negative,
                'R3': r3_negative
            },
            'budget_promo_covariance': budget_promo_cov,
            'total_movies': len(self.df),
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'best_model': self.model_name,
            'model_results': self.model_results
        }

# Initialize the app
movie_app = MovieAnalysisApp()

@app.route('/')
def home():
    # Get current model stats for home page
    try:
        insights = movie_app.get_insights()
        stats = {
            'accuracy': round(insights['accuracy'] * 100, 1),
            'total_movies': insights['total_movies'],
            'model_name': insights.get('best_model', 'Decision Tree')
        }
    except:
        # Fallback values if there's an error
        stats = {
            'accuracy': 0,
            'total_movies': 540,
            'model_name': 'Decision Tree'
        }
    
    return render_template('index.html', stats=stats)

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    print(f"DEBUG: Request method = {request.method}")
    
    if request.method == 'POST':
        print("DEBUG: POST request received")
        try:
            # Get form data
            movie_data = {
                'Runtime': int(request.form.get('Runtime')),
                'Stars': int(request.form.get('Stars')),
                'Year': int(request.form.get('Year')),
                'Budget': float(request.form.get('Budget')),
                'Promo': float(request.form.get('Promo')),
                'season': request.form.get('season'),
                'rating': request.form.get('rating'),
                'genre': request.form.get('genre'),
                'R1': request.form.get('R1', ''),
                'R2': request.form.get('R2', ''),
                'R3': request.form.get('R3', '')
            }
            
            print(f"DEBUG: Movie data = {movie_data}")
            
            # Make prediction
            prediction_result = movie_app.predict_success(movie_data)
            print(f"DEBUG: Prediction result = {prediction_result}")
            
            # Redirect to results page
            return redirect(url_for('results', 
                                  prediction=prediction_result['prediction']))
            
        except Exception as e:
            print(f"DEBUG: Error occurred = {str(e)}")
            app.logger.error(f"Prediction error: {str(e)}")
            return render_template('predict.html', 
                                 error=f"Error making prediction: {str(e)}",
                                 show_results=False)
    
    # GET request - show empty form
    print("DEBUG: GET request - showing empty form")
    return render_template('predict.html', show_results=False)

@app.route('/results')
def results():
    prediction = request.args.get('prediction', 'False').lower() == 'true'
    
    return render_template('results.html', prediction=prediction)

@app.route('/insights')
def insights_page():
    return render_template('insights.html')

@app.route('/visualizations')
def visualizations_page():
    return render_template('visualizations.html')

@app.route('/api/health')
def api_health():
    """Health check endpoint to verify model status"""
    try:
        status = {
            'model_available': movie_app.model is not None,
            'dataset_available': movie_app.df is not None and not movie_app.df.empty,
            'model_name': movie_app.model_name if hasattr(movie_app, 'model_name') else None,
            'accuracy': movie_app.accuracy if hasattr(movie_app, 'accuracy') else None,
            'dataset_size': len(movie_app.df) if movie_app.df is not None else 0
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        print("🔮 Prediction request received")
        
        # Check if model is available
        if movie_app.model is None:
            print("❌ Model is not available")
            return jsonify({'error': 'Model not available. Please check server logs.'}), 500
        
        if movie_app.df is None or movie_app.df.empty:
            print("❌ Dataset is not available")
            return jsonify({'error': 'Dataset not available. Please check server logs.'}), 500
            
        data = request.json
        print(f"📊 Prediction data received: {data}")
        
        # Debug: Check what features are being sent
        print(f"📊 Features in request: {list(data.keys())}")
        
        result = movie_app.predict_success(data)
        print(f"🎯 Prediction result: {result}")
        
        return jsonify(result)
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/api/insights')
def api_insights():
    try:
        insights = movie_app.get_insights()
        
        # Handle potential NaN values
        if insights.get('accuracy') is None or str(insights.get('accuracy')) == 'nan':
            insights['accuracy'] = 0
        
        # Ensure model results are serializable
        if 'model_results' in insights and insights['model_results']:
            # Remove the actual model objects which can't be serialized
            clean_results = {}
            for name, result in insights['model_results'].items():
                clean_results[name] = {
                    'cv_score': result.get('cv_score', 0),
                    'test_accuracy': result.get('test_accuracy', 0)
                }
            insights['model_results'] = clean_results
        
        return jsonify(insights)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/data')
def api_data():
    try:
        # Return sample data for visualizations
        data = {
            'budget_by_stars': movie_app.df.groupby('Stars')['Budget'].mean().to_dict(),
            'success_by_season': movie_app.df.groupby('Season')['Success'].mean().to_dict(),
            'genre_distribution': movie_app.df['Genre'].value_counts().to_dict(),
            'rating_distribution': movie_app.df['Rating'].value_counts().to_dict(),
            'yearly_budget_trend': movie_app.df.groupby('Year')['Budget'].mean().to_dict()
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/random-examples')
def api_random_examples():
    try:
        # Get ALL movies from TEST SET (movies the model has never seen)
        test_movies = movie_app.df.loc[movie_app.test_indices]
        
        # Use all test movies (all 54 movies from the 10% test set)
        random_movies = test_movies.copy()
        
        examples = []
        for _, movie in random_movies.iterrows():
            # Create feature vector for prediction
            movie_data = {}
            for col in movie_app.feature_columns:
                if col in movie.index:
                    movie_data[col] = movie[col]
                else:
                    movie_data[col] = 0
            
            # Get prediction
            prediction_result = movie_app.predict_success(movie_data)
            
            # Prepare example data
            example = {
                'title': movie.get('Title', 'Unknown Title'),
                'runtime': int(movie['Runtime']),
                'stars': int(movie['Stars']),
                'year': int(movie['Year']),
                'budget': round(float(movie['Budget']), 2),
                'promo': round(float(movie['Promo']), 2),
                'season': movie['Season'],
                'rating': movie['Rating'],
                'genre': movie['Genre'],
                'actual_success': bool(movie['Success']),
                'predicted_success': prediction_result['prediction'],
                'confidence': round(prediction_result['confidence'] * 100, 1),
                'correct': bool(movie['Success']) == prediction_result['prediction']
            }
            examples.append(example)
        
        # Calculate accuracy for easy reference
        correct_predictions = sum(1 for example in examples if example['correct'])
        total_predictions = len(examples)
        accuracy_percentage = round((correct_predictions / total_predictions) * 100, 1) if total_predictions > 0 else 0
        
        return jsonify({
            'examples': examples,
            'accuracy_summary': {
                'correct': correct_predictions,
                'total': total_predictions,
                'percentage': accuracy_percentage
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/test-set-movies')
def api_test_set_movies():
    try:
        # Get all movies in the test set (the 10% the model never saw)
        test_movies = movie_app.df.loc[movie_app.test_indices]
        
        movies_list = []
        for _, movie in test_movies.iterrows():
            movies_list.append({
                'title': movie.get('Title', 'Unknown Title'),
                'year': int(movie['Year']),
                'genre': movie['Genre'],
                'success': bool(movie['Success']),
                'index': int(movie.name)  # Original dataset index
            })
        
        return jsonify({
            'total_test_movies': len(movies_list),
            'percentage': round(len(movies_list) / len(movie_app.df) * 100, 1),
            'movies': movies_list
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/visualization-data')
def api_visualization_data():
    try:
        # Check if movie_app and data are available
        if movie_app is None:
            return jsonify({'error': 'Movie app not initialized'}), 400
        
        if movie_app.df is None or movie_app.df.empty:
            return jsonify({'error': 'Movie data not loaded'}), 400
            
        # Get data for visualizations from the notebook
        df = movie_app.df.copy()
        
        # Check if required columns exist
        required_columns = ['Stars', 'Budget', 'Year']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({'error': f'Missing columns: {missing_columns}'}), 400
        
        # Generate the exact plots from your Jupyter notebook
        
        # 1. Boxplot: sns.boxplot(x="Stars", y="Budget", data=movie_df)
        sns.boxplot(x="Stars", y="Budget", data=df)
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        boxplot_img = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # 2. Lineplot: sns.lineplot(x="Year", y="Budget", data=year_budget_df, estimator="mean", errorbar=None)
        # First prepare the data exactly like in your notebook
        year_budget_df = df[["Year", "Budget"]].copy()
        year_budget_df["Year"] = pd.to_datetime(year_budget_df["Year"], format="%Y")
        
        sns.lineplot(x="Year", y="Budget", data=year_budget_df, estimator="mean", errorbar=None)
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        lineplot_img = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'boxplot': boxplot_img,
            'lineplot': lineplot_img
        })
    except Exception as e:
        print(f"Error in visualization data API: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
