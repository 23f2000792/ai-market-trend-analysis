import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)

# Advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available, using sklearn models only")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    A comprehensive model training class that:
    1. Prepares data for machine learning
    2. Trains multiple models
    3. Evaluates and compares performance
    4. Saves the best model for deployment
    """

    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.feature_importance = {}

        print("🤖 ModelTrainer initialized!")
        print("   Ready to train and evaluate ML models")

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning.

        Args:
            df: DataFrame with features and target
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\n🔧 Preparing data for machine learning...")
        print("=" * 45)

        # Remove rows with missing target
        df_clean = df.dropna(subset=['Target']).copy()

        # Define feature columns (exclude non-feature columns)
        exclude_cols = [
            'Date', 'Symbol', 'Target', 'Target_Binary', 'Next_Day_Return',
            'Open', 'High', 'Low', 'Close', 'Volume'  # Keep raw price data separate
        ]

        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

        # Prepare features and target
        X = df_clean[feature_cols].copy()
        y = df_clean['Target'].copy()

        print(f"📊 Dataset preparation:")
        print(f"   📈 Total samples: {len(X):,}")
        print(f"   🔧 Features: {len(feature_cols)}")
        print(f"   🎯 Target classes: {sorted(y.unique())}")

        # Show class distribution
        class_dist = y.value_counts().sort_index()
        print(f"   📊 Class distribution:")
        for class_val, count in class_dist.items():
            class_name = ['Down', 'Stable', 'Up'][class_val]
            print(f"      {class_name} ({class_val}): {count:,} ({count/len(y)*100:.1f}%)")

        # Handle any remaining missing values
        if X.isnull().sum().sum() > 0:
            print("   🔧 Handling missing values...")
            X = X.fillna(X.median())

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y  # Maintain class distribution
        )

        # Scale the features
        print("   📏 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\n✅ Data preparation complete!")
        print(f"   🏋️  Training set: {len(X_train):,} samples")
        print(f"   🧪 Test set: {len(X_test):,} samples")

        # Store feature names for later use
        self.feature_names = feature_cols

        return X_train_scaled, X_test_scaled, y_train, y_test

    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all available models with good default parameters.

        Returns:
            Dictionary of initialized models
        """
        print("\n🤖 Initializing models...")

        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss'
            )
            print("   ✅ XGBoost added")

        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            print("   ✅ LightGBM added")

        print(f"   🎯 Total models: {len(models)}")
        return models

    def train_model(self, model, X_train: np.ndarray, y_train: np.ndarray, 
                   model_name: str) -> Dict[str, Any]:
        """
        Train a single model and return training metrics.

        Args:
            model: Sklearn-compatible model
            X_train: Training features
            y_train: Training targets
            model_name: Name of the model

        Returns:
            Dictionary with training results
        """
        print(f"   🏋️  Training {model_name}...")

        # Train the model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # Cross-validation score
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, 
            scoring='accuracy', n_jobs=-1
        )

        results = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time
        }

        print(f"      ✅ CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"      ⏱️  Training time: {training_time:.2f}s")

        return results

    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model

        Returns:
            Dictionary with evaluation results
        """
        print(f"\n📊 Evaluating {model_name}...")

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # ROC AUC (for multiclass)
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else None
        except:
            auc_score = None

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }

        print(f"   📈 Test Results:")
        print(f"      Accuracy:  {accuracy:.3f}")
        print(f"      Precision: {precision:.3f}")
        print(f"      Recall:    {recall:.3f}")
        print(f"      F1-Score:  {f1:.3f}")
        if auc_score:
            print(f"      AUC Score: {auc_score:.3f}")

        return results

    def get_feature_importance(self, model, model_name: str) -> pd.DataFrame:
        """
        Extract feature importance from trained model.

        Args:
            model: Trained model
            model_name: Name of the model

        Returns:
            DataFrame with feature importance
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                importance = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
            else:
                return pd.DataFrame()

            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)

            return importance_df

        except Exception as e:
            print(f"   ⚠️  Could not extract feature importance: {e}")
            return pd.DataFrame()

    def train_all_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                        y_train: np.ndarray, y_test: np.ndarray) -> None:
        """
        Train and evaluate all available models.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
        """
        print("\n🚀 Training all models...")
        print("=" * 40)

        # Initialize models
        models = self.initialize_models()

        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"\n🔄 Processing {model_name}...")

            # Train the model
            train_results = self.train_model(model, X_train, y_train, model_name)

            # Evaluate on test set
            eval_results = self.evaluate_model(model, X_test, y_test, model_name)

            # Get feature importance
            importance_df = self.get_feature_importance(model, model_name)

            # Store results
            self.models[model_name] = train_results['model']
            self.results[model_name] = {
                **train_results,
                **eval_results,
                'feature_importance': importance_df
            }

            # Update best model
            if eval_results['accuracy'] > self.best_score:
                self.best_score = eval_results['accuracy']
                self.best_model = train_results['model']
                self.best_model_name = model_name
                print(f"   🏆 New best model: {model_name} (Accuracy: {self.best_score:.3f})")

        self._print_model_comparison()

    def _print_model_comparison(self) -> None:
        """Print a comparison of all trained models."""
        print("\n" + "=" * 60)
        print("🏆 MODEL COMPARISON RESULTS")
        print("=" * 60)

        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.3f}",
                'F1-Score': f"{results['f1_score']:.3f}",
                'CV Score': f"{results['cv_mean']:.3f} ± {results['cv_std']:.3f}",
                'Training Time': f"{results['training_time']:.2f}s"
            })

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        print(f"\n🥇 Best Model: {self.best_model_name} (Accuracy: {self.best_score:.3f})")

    def save_models(self, models_dir: str = "models") -> None:
        """
        Save trained models and metadata.

        Args:
            models_dir: Directory to save models
        """
        print(f"\n💾 Saving models to {models_dir}/...")

        # Create models directory
        os.makedirs(models_dir, exist_ok=True)

        # Save each model
        for model_name, model in self.models.items():
            model_filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
            model_path = os.path.join(models_dir, model_filename)
            joblib.dump(model, model_path)
            print(f"   ✅ Saved {model_name} to {model_path}")

        # Save scaler
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"   ✅ Saved scaler to {scaler_path}")

        # Save metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'best_model_accuracy': self.best_score,
            'feature_names': self.feature_names,
            'model_results': {
                name: {
                    'accuracy': float(results['accuracy']),
                    'f1_score': float(results['f1_score']),
                    'cv_mean': float(results['cv_mean']),
                    'cv_std': float(results['cv_std']),
                    'training_time': float(results['training_time'])
                }
                for name, results in self.results.items()
            },
            'training_date': datetime.now().isoformat()
        }

        metadata_path = os.path.join(models_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Saved metadata to {metadata_path}")

        # Save feature importance for best model
        if self.best_model_name in self.results:
            importance_df = self.results[self.best_model_name]['feature_importance']
            if not importance_df.empty:
                importance_path = os.path.join(models_dir, "feature_importance.csv")
                importance_df.to_csv(importance_path, index=False)
                print(f"   ✅ Saved feature importance to {importance_path}")

    def load_models(self, models_dir: str = "models") -> None:
        """
        Load saved models and metadata.

        Args:
            models_dir: Directory containing saved models
        """
        print(f"📂 Loading models from {models_dir}/...")

        # Load metadata
        metadata_path = os.path.join(models_dir, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.best_model_name = metadata['best_model_name']
            self.best_score = metadata['best_model_accuracy']
            self.feature_names = metadata['feature_names']
            print(f"   ✅ Best model: {self.best_model_name} (Accuracy: {self.best_score:.3f})")

        # Load scaler
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"   ✅ Loaded scaler")

        # Load best model
        if self.best_model_name:
            model_filename = f"{self.best_model_name.lower().replace(' ', '_')}_model.pkl"
            model_path = os.path.join(models_dir, model_filename)
            if os.path.exists(model_path):
                self.best_model = joblib.load(model_path)
                print(f"   ✅ Loaded {self.best_model_name}")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the best model.

        Args:
            X: Features to predict on

        Returns:
            predictions, prediction_probabilities
        """
        if self.best_model is None:
            raise ValueError("No model available. Train a model first or load saved models.")

        X_scaled = self.scaler.transform(X)
        predictions = self.best_model.predict(X_scaled)
        probabilities = self.best_model.predict_proba(X_scaled) if hasattr(self.best_model, 'predict_proba') else None

        return predictions, probabilities


def main():
    """
    Example usage of the ModelTrainer.
    This demonstrates the complete model training pipeline.
    """
    print("🚀 AI Market Trend Analysis - Model Training Demo")
    print("=" * 60)

    try:
        # Load processed data
        print("📂 Loading processed stock data...")
        df = pd.read_csv("data/features/stock_features.csv")
        df['Date'] = pd.to_datetime(df['Date'])

        print(f"   ✅ Loaded {len(df):,} rows of processed data")

        # Initialize trainer
        trainer = ModelTrainer()

        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(df, test_size=0.2)

        # Train all models
        trainer.train_all_models(X_train, X_test, y_train, y_test)

        # Save models
        trainer.save_models()

        # Show best model details
        print("\n🔍 Best Model Details:")
        if trainer.best_model_name in trainer.results:
            results = trainer.results[trainer.best_model_name]

            # Classification report
            print("\n📊 Detailed Classification Report:")
            y_pred = results['predictions']
            report = classification_report(y_test, y_pred, 
                                         target_names=['Down', 'Stable', 'Up'])
            print(report)

            # Feature importance top 10
            importance_df = results['feature_importance']
            if not importance_df.empty:
                print("\n🔧 Top 10 Most Important Features:")
                for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                    print(f"   {i:2d}. {row['Feature']}: {row['Importance']:.4f}")

        print("\n🎉 Model training complete! Models saved and ready for deployment.")

    except FileNotFoundError:
        print("❌ Error: stock_features.csv not found!")
        print("   Please run feature_engineer.py first to create the processed features.")
    except Exception as e:
        print(f"❌ Error during model training: {e}")


if __name__ == "__main__":
    main()