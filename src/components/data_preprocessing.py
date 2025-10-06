import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class FinancialNewsPreprocessor:
    """
    Comprehensive preprocessing pipeline for financial news sentiment analysis
    """
    
    def __init__(self, data_path='data\all_data.csv'):
        """
        Initialize preprocessor with data path
        
        Args:
            data_path: Path to the CSV file containing financial news data
        """
        self.data_path = data_path
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.label_encoder = LabelEncoder()
        
        # Financial domain-specific stopwords to keep (they carry sentiment)
        self.financial_keywords = {
            'profit', 'loss', 'gain', 'fall', 'rise', 'growth', 'decline',
            'increase', 'decrease', 'revenue', 'earnings', 'debt', 'equity'
        }
        self.stop_words = self.stop_words - self.financial_keywords

    def load_data(self):
        """
        Load and perform initial data inspection
        
        Returns:
            DataFrame: Loaded dataset
        """
        print("=" * 60)
        print("STEP 1: LOADING DATA")
        print("=" * 60)
        
        try:
            df = pd.read_csv(self.data_path, encoding='ISO-8859-1', 
                           names=['sentiment', 'text'], header=None)
            print(f"✓ Data loaded successfully")
            print(f"  - Total samples: {len(df)}")
            print(f"  - Columns: {list(df.columns)}")
            print(f"\nFirst few rows:")
            print(df.head())
            print(f"\nSentiment distribution:")
            print(df['sentiment'].value_counts())
            print(f"\nData types:")
            print(df.dtypes)
            
            return df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise

    def check_missing_duplicates(self, df):
        """
        Check for missing values and duplicates
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Cleaned DataFrame
        """
        print("\n" + "=" * 60)
        print("STEP 2: DATA QUALITY CHECK")
        print("=" * 60)
        
        # Check missing values
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
        # Remove rows with missing values
        initial_count = len(df)
        df = df.dropna()
        removed_missing = initial_count - len(df)
        print(f"✓ Removed {removed_missing} rows with missing values")
        
        # Check duplicates
        duplicate_count = df.duplicated().sum()
        print(f"\nDuplicate rows found: {duplicate_count}")
        df = df.drop_duplicates()
        print(f"✓ Removed {duplicate_count} duplicate rows")
        
        print(f"\nFinal dataset size: {len(df)} rows")
        
        return df
    
    def clean_text(self, text):
        """
        Clean individual text entry
        
        Args:
            text: Input text string
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    

    def tokenize_and_lemmatize(self, text):
        """
        Tokenize and lemmatize text
        
        Args:
            text: Input text string
            
        Returns:
            str: Processed text
        """
        if not text:
            return ""
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(processed_tokens)
    

    def preprocess_text(self, df):
        """
        Apply all text preprocessing steps
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: DataFrame with preprocessed text
        """
        print("\n" + "=" * 60)
        print("STEP 3: TEXT PREPROCESSING")
        print("=" * 60)
        
        # Clean text
        print("\n⊳ Cleaning text (removing URLs, special characters, etc.)...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        print("✓ Text cleaning completed")
        
        # Tokenize and lemmatize
        print("\n⊳ Tokenizing and lemmatizing...")
        df['processed_text'] = df['cleaned_text'].apply(self.tokenize_and_lemmatize)
        print("✓ Tokenization and lemmatization completed")
        
        # Remove empty processed texts
        initial_count = len(df)
        df = df[df['processed_text'].str.len() > 0]
        removed_empty = initial_count - len(df)
        print(f"\n✓ Removed {removed_empty} rows with empty processed text")
        
        # Display examples
        print("\n" + "-" * 60)
        print("Example transformations:")
        print("-" * 60)
        for idx in range(min(3, len(df))):
            print(f"\nOriginal: {df.iloc[idx]['text'][:100]}...")
            print(f"Processed: {df.iloc[idx]['processed_text'][:100]}...")
        
        return df
    

    def encode_labels(self, df):
        """
        Encode sentiment labels to numerical values
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: DataFrame with encoded labels
        """
        print("\n" + "=" * 60)
        print("STEP 4: LABEL ENCODING")
        print("=" * 60)
        
        # Encode labels
        df['sentiment_encoded'] = self.label_encoder.fit_transform(df['sentiment'])
        
        # Display mapping
        print("\nSentiment encoding mapping:")
        for i, label in enumerate(self.label_encoder.classes_):
            count = (df['sentiment_encoded'] == i).sum()
            print(f"  {label} → {i} ({count} samples)")
        
        return df
    

    def get_text_statistics(self, df):
        """
        Calculate and display text statistics
        
        Args:
            df: Input DataFrame
        """
        print("\n" + "=" * 60)
        print("STEP 5: TEXT STATISTICS")
        print("=" * 60)
        
        # Calculate statistics
        df['text_length'] = df['processed_text'].str.split().str.len()
        
        print(f"\nText length statistics (in words):")
        print(f"  Mean: {df['text_length'].mean():.2f}")
        print(f"  Median: {df['text_length'].median():.2f}")
        print(f"  Min: {df['text_length'].min()}")
        print(f"  Max: {df['text_length'].max()}")
        print(f"  Std: {df['text_length'].std():.2f}")
        
        # Statistics by sentiment
        print(f"\nAverage text length by sentiment:")
        for sentiment in df['sentiment'].unique():
            avg_len = df[df['sentiment'] == sentiment]['text_length'].mean()
            print(f"  {sentiment}: {avg_len:.2f} words")

    
    def split_data(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            test_size: Proportion of test set
            val_size: Proportion of validation set from training data
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n" + "=" * 60)
        print("STEP 6: DATA SPLITTING")
        print("=" * 60)
        
        X = df['processed_text'].values
        y = df['sentiment_encoded'].values
        
        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Split train+val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"\nData split completed:")
        print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Check class distribution
        print(f"\nClass distribution in splits:")
        for split_name, split_y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            unique, counts = np.unique(split_y, return_counts=True)
            dist = dict(zip(unique, counts))
            print(f"  {split_name}: {dist}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                              output_dir='preprocessed_data'):
        """
        Save preprocessed data and metadata
        
        Args:
            X_train, X_val, X_test: Feature arrays
            y_train, y_val, y_test: Label arrays
            output_dir: Directory to save preprocessed data
        """
        print("\n" + "=" * 60)
        print("STEP 7: SAVING PREPROCESSED DATA")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save splits
        np.save(f'{output_dir}/X_train.npy', X_train)
        np.save(f'{output_dir}/X_val.npy', X_val)
        np.save(f'{output_dir}/X_test.npy', X_test)
        np.save(f'{output_dir}/y_train.npy', y_train)
        np.save(f'{output_dir}/y_val.npy', y_val)
        np.save(f'{output_dir}/y_test.npy', y_test)
        
        # Save label encoder
        with open(f'{output_dir}/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"\n✓ All preprocessed data saved to '{output_dir}/' directory")
        print(f"  Files created:")
        print(f"    - X_train.npy, X_val.npy, X_test.npy")
        print(f"    - y_train.npy, y_val.npy, y_test.npy")
        print(f"    - label_encoder.pkl")
    
    def run_pipeline(self):
        """
        Execute the complete preprocessing pipeline
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n" + "█" * 60)
        print("FINANCIAL NEWS SENTIMENT ANALYSIS - PREPROCESSING PIPELINE")
        print("█" * 60)
        
        # Load data
        df = self.load_data()
        
        # Check quality
        df = self.check_missing_duplicates(df)
        
        # Preprocess text
        df = self.preprocess_text(df)
        
        # Encode labels
        df = self.encode_labels(df)
        
        # Get statistics
        self.get_text_statistics(df)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)
        
        # Save data
        self.save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test)
        
        print("\n" + "█" * 60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("█" * 60)
        print("\nYou can now proceed to model training using 'model_training.py'")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


# Main execution
if __name__ == "__main__":
    # Initialize preprocessor
    # Make sure to update the data_path to your actual file path
    preprocessor = FinancialNewsPreprocessor(data_path='data.csv')
    
    # Run the complete pipeline
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.run_pipeline()