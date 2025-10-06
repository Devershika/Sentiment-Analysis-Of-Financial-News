import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            f1_score, precision_score, recall_score, roc_auc_score,
                            roc_curve)
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
class FinancialSentimentTrainer:
    """
    Comprehensive model training and evaluation pipeline for financial sentiment analysis
    """
    
    def __init__(self, data_dir='preprocessed_data'):
        """
        Initialize trainer with preprocessed data directory
        
        Args:
            data_dir: Directory containing preprocessed data
        """
        self.data_dir = data_dir
        self.vectorizer = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_preprocessed_data(self):
        """
        Load preprocessed data from saved files
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test, label_encoder)
        """
        print("=" * 70)
        print("STEP 1: LOADING PREPROCESSED DATA")
        print("=" * 70)
        
        try:
            X_train = np.load(f'{self.data_dir}/X_train.npy', allow_pickle=True)
            X_val = np.load(f'{self.data_dir}/X_val.npy', allow_pickle=True)
            X_test = np.load(f'{self.data_dir}/X_test.npy', allow_pickle=True)
            y_train = np.load(f'{self.data_dir}/y_train.npy', allow_pickle=True)
            y_val = np.load(f'{self.data_dir}/y_val.npy', allow_pickle=True)
            y_test = np.load(f'{self.data_dir}/y_test.npy', allow_pickle=True)
            
            with open(f'{self.data_dir}/label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
            
            print(f"✓ Data loaded successfully")
            print(f"  Training samples: {len(X_train)}")
            print(f"  Validation samples: {len(X_val)}")
            print(f"  Test samples: {len(X_test)}")
            print(f"  Number of classes: {len(label_encoder.classes_)}")
            print(f"  Classes: {list(label_encoder.classes_)}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def create_features(self, X_train, X_val, X_test, method='tfidf', max_features=5000):
        """
        Create features using TF-IDF or Count Vectorization
        
        Args:
            X_train, X_val, X_test: Text data arrays
            method: 'tfidf' or 'count'
            max_features: Maximum number of features
            
        Returns:
            tuple: (X_train_vec, X_val_vec, X_test_vec)
        """
        print("\n" + "=" * 70)
        print("STEP 2: FEATURE EXTRACTION")
        print("=" * 70)
        
        if method == 'tfidf':
            print(f"\n⊳ Using TF-IDF Vectorization (max_features={max_features})...")
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),  # Unigrams and bigrams
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        else:
            print(f"\n⊳ Using Count Vectorization (max_features={max_features})...")
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        
        # Fit and transform
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_val_vec = self.vectorizer.transform(X_val)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"✓ Feature extraction completed")
        print(f"  Training set shape: {X_train_vec.shape}")
        print(f"  Validation set shape: {X_val_vec.shape}")
        print(f"  Test set shape: {X_test_vec.shape}")
        print(f"  Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
        
        return X_train_vec, X_val_vec, X_test_vec
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """
        Train multiple models and compare performance
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        print("\n" + "=" * 70)
        print("STEP 3: MODEL TRAINING")
        print("=" * 70)
        
        # Define models
        models_config = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Linear SVM': LinearSVC(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        print(f"\nTraining {len(models_config)} models...\n")
        
        for name, model in models_config.items():
            print(f"⊳ Training {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Calculate metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')
            
            # Store model and results
            self.models[name] = model
            self.results[name] = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'val_f1': val_f1,
                'y_val_pred': y_val_pred
            }
            
            print(f"  ✓ Train Accuracy: {train_acc:.4f}")
            print(f"  ✓ Val Accuracy: {val_acc:.4f}")
            print(f"  ✓ Val F1-Score: {val_f1:.4f}\n")
    
    def compare_models(self):
        """
        Compare all trained models and select the best one
        """
        print("=" * 70)
        print("STEP 4: MODEL COMPARISON")
        print("=" * 70)
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Train Accuracy': [self.results[m]['train_accuracy'] for m in self.results],
            'Val Accuracy': [self.results[m]['val_accuracy'] for m in self.results],
            'Val F1-Score': [self.results[m]['val_f1'] for m in self.results]
        })
        
        # Sort by validation accuracy
        comparison = comparison.sort_values('Val Accuracy', ascending=False)
        
        print("\nModel Performance Comparison:")
        print("-" * 70)
        print(comparison.to_string(index=False))
        print("-" * 70)
        
        # Select best model
        self.best_model_name = comparison.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"   Validation Accuracy: {comparison.iloc[0]['Val Accuracy']:.4f}")
        print(f"   Validation F1-Score: {comparison.iloc[0]['Val F1-Score']:.4f}")
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        """
        Perform hyperparameter tuning on the best model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        print("\n" + "=" * 70)
        print("STEP 5: HYPERPARAMETER TUNING")
        print("=" * 70)
        
        print(f"\n⊳ Tuning {self.best_model_name}...")
        
        # Define parameter grids for different models
        param_grids = {
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
            'Linear SVM': {
                'C': [0.1, 1, 10],
                'loss': ['hinge', 'squared_hinge']
            },
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        }
        
        if self.best_model_name in param_grids:
            param_grid = param_grids[self.best_model_name]
            
            # Combine train and val for GridSearch
            X_combined = np.vstack([X_train.toarray(), X_val.toarray()])
            y_combined = np.concatenate([y_train, y_val])
            
            grid_search = GridSearchCV(
                self.best_model,
                param_grid,
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_combined, y_combined)
            
            self.best_model = grid_search.best_estimator_
            self.models[self.best_model_name] = self.best_model
            
            print(f"\n✓ Best parameters: {grid_search.best_params_}")
            print(f"✓ Best cross-validation F1-score: {grid_search.best_score_:.4f}")
        else:
            print(f"  No hyperparameter tuning configured for {self.best_model_name}")
            print(f"  Using default parameters")
    
    def evaluate_on_test(self, X_test, y_test, label_encoder):
        """
        Final evaluation on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            label_encoder: Label encoder for class names
        """
        print("\n" + "=" * 70)
        print("STEP 6: FINAL EVALUATION ON TEST SET")
        print("=" * 70)
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n{self.best_model_name} - Test Set Performance:")
        print("-" * 70)
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print("-" * 70)
        
        # Classification report
        print("\nDetailed Classification Report:")
        print("-" * 70)
        print(classification_report(y_test, y_pred, 
                                   target_names=label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, label_encoder.classes_)
        
        return accuracy, precision, recall, f1
    
    def plot_confusion_matrix(self, cm, classes):
        """
        Plot confusion matrix heatmap
        
        Args:
            cm: Confusion matrix
            classes: Class names
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {self.best_model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
        plt.close()
    
    def plot_model_comparison(self):
        """
        Plot model comparison bar chart
        """
        models = list(self.results.keys())
        train_accs = [self.results[m]['train_accuracy'] for m in models]
        val_accs = [self.results[m]['val_accuracy'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
        ax.bar(x + width/2, val_accs, width, label='Val Accuracy', alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Model comparison chart saved as 'model_comparison.png'")
        plt.close()
    
    def save_model(self, output_dir='trained_models'):
        """
        Save trained model and vectorizer
        
        Args:
            output_dir: Directory to save models
        """
        print("\n" + "=" * 70)
        print("STEP 7: SAVING TRAINED MODEL")
        print("=" * 70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = f'{output_dir}/best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save vectorizer
        vectorizer_path = f'{output_dir}/vectorizer.pkl'
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save model info
        info = {
            'model_name': self.best_model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': self.results
        }
        
        info_path = f'{output_dir}/model_info.pkl'
        with open(info_path, 'wb') as f:
            pickle.dump(info, f)
        
        print(f"\n✓ Model saved successfully to '{output_dir}/' directory")
        print(f"  Files created:")
        print(f"    - best_model.pkl ({self.best_model_name})")
        print(f"    - vectorizer.pkl")
        print(f"    - model_info.pkl")
    
    def predict_sentiment(self, texts, label_encoder):
        """
        Predict sentiment for new texts
        
        Args:
            texts: List of text strings or single text string
            label_encoder: Label encoder for decoding predictions
            
        Returns:
            predictions and probabilities
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Vectorize
        X = self.vectorizer.transform(texts)
        
        # Predict
        predictions = self.best_model.predict(X)
        
        # Get probabilities if available
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(X)
        else:
            probabilities = None
        
        # Decode labels
        predicted_labels = label_encoder.inverse_transform(predictions)
        
        return predicted_labels, probabilities
    
    def run_pipeline(self, tune_hyperparameters=True):
        """
        Execute the complete training pipeline
        
        Args:
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        print("\n" + "█" * 70)
        print("FINANCIAL NEWS SENTIMENT ANALYSIS - MODEL TRAINING PIPELINE")
        print("█" * 70)
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = \
            self.load_preprocessed_data()
        
        # Create features
        X_train_vec, X_val_vec, X_test_vec = self.create_features(
            X_train, X_val, X_test, method='tfidf'
        )
        
        # Train models
        self.train_models(X_train_vec, y_train, X_val_vec, y_val)
        
        # Compare models
        self.compare_models()
        
        # Plot comparison
        self.plot_model_comparison()
        
        # Hyperparameter tuning
        if tune_hyperparameters:
            self.hyperparameter_tuning(X_train_vec, y_train, X_val_vec, y_val)
        
        # Final evaluation
        accuracy, precision, recall, f1 = self.evaluate_on_test(
            X_test_vec, y_test, label_encoder
        )
        
        # Save model
        self.save_model()
        
        print("\n" + "=" * 70)
        print("EXAMPLE PREDICTIONS")
        print("=" * 70)
        
        # Example predictions
        example_texts = [
            "Company reports record profits and strong revenue growth",
            "Stock prices plummet amid bankruptcy fears",
            "Quarterly earnings meet analyst expectations"
        ]
        
        predictions, probabilities = self.predict_sentiment(example_texts, label_encoder)
        
        print("\nSample predictions on new texts:")
        print("-" * 70)
        for i, text in enumerate(example_texts):
            print(f"\nText: {text}")
            print(f"Predicted Sentiment: {predictions[i]}")
            if probabilities is not None:
                print(f"Confidence scores:")
                for j, label in enumerate(label_encoder.classes_):
                    print(f"  {label}: {probabilities[i][j]:.4f}")
        
        print("\n" + "█" * 70)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("█" * 70)
        print(f"\nFinal Test Metrics:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"\nBest Model: {self.best_model_name}")
        print(f"\nModel saved and ready for deployment!")


class SentimentPredictor:
    """
    Standalone predictor class for loading and using trained model
    """
    
    def __init__(self, model_dir='trained_models', preprocessed_data_dir='preprocessed_data'):
        """
        Initialize predictor by loading saved model
        
        Args:
            model_dir: Directory containing trained model
            preprocessed_data_dir: Directory containing label encoder
        """
        self.model_dir = model_dir
        self.preprocessed_data_dir = preprocessed_data_dir
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.load_model()
    
    def load_model(self):
        """Load trained model, vectorizer, and label encoder"""
        try:
            # Load model
            with open(f'{self.model_dir}/best_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Load vectorizer
            with open(f'{self.model_dir}/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load label encoder
            with open(f'{self.preprocessed_data_dir}/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            print("✓ Model loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def predict(self, texts):
        """
        Predict sentiment for new texts
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        # Vectorize
        X = self.vectorizer.transform(texts)
        
        # Predict
        predictions = self.model.predict(X)
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            
            results = []
            for i, text in enumerate(texts):
                prob_dict = {
                    label: float(probabilities[i][j])
                    for j, label in enumerate(self.label_encoder.classes_)
                }
                results.append({
                    'text': text,
                    'predicted_sentiment': predicted_labels[i],
                    'confidence_scores': prob_dict,
                    'confidence': float(max(probabilities[i]))
                })
        else:
            results = []
            for i, text in enumerate(texts):
                results.append({
                    'text': text,
                    'predicted_sentiment': predicted_labels[i]
                })
        
        return results[0] if single_input else results


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FINANCIAL SENTIMENT ANALYSIS - MODEL TRAINING")
    print("=" * 70)
    print("\nThis script will train multiple models and select the best one.")
    print("Make sure you have run data_preprocessing.py first!\n")
    
    # Initialize trainer
    trainer = FinancialSentimentTrainer(data_dir='preprocessed_data')
    
    # Run the complete pipeline
    trainer.run_pipeline(tune_hyperparameters=True)
    
    print("\n" + "=" * 70)
    print("USING THE TRAINED MODEL")
    print("=" * 70)
    print("\nTo use the trained model for predictions, you can:")
    print("\n1. Use the SentimentPredictor class:")
    print("   predictor = SentimentPredictor()")
    print("   result = predictor.predict('Your financial news text here')")
    print("\n2. Or load the model manually:")
    print("   with open('trained_models/best_model.pkl', 'rb') as f:")
    print("       model = pickle.load(f)")
    
    # Demonstrate usage
    print("\n" + "=" * 70)
    print("DEMONSTRATION: USING SENTIMENTPREDICTOR")
    print("=" * 70)
    
    try:
        predictor = SentimentPredictor()
        
        demo_texts = [
            "The company announced record-breaking quarterly profits",
            "Shares dropped 15% following disappointing earnings report",
            "Market analysts maintain neutral stance on the stock"
        ]
        
        print("\nPredicting sentiment for demo texts:\n")
        results = predictor.predict(demo_texts)
        
        for result in results:
            print(f"Text: {result['text']}")
            print(f"Sentiment: {result['predicted_sentiment']}")
            if 'confidence_scores' in result:
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"All scores: {result['confidence_scores']}")
            print("-" * 70)
            
    except Exception as e:
        print(f"\nNote: Predictor demo skipped - {e}")
        print("Run the training pipeline first to generate the model.")