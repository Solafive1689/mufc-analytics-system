"""
Machine Learning Models for Manchester United Analytics
Predictive models for match outcomes, player performance, and tactical insights
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import joblib
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MatchPredictor:
    """Predict match outcomes based on historical data"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def prepare_features(self, lookback_games: int = 5) -> pd.DataFrame:
        """Prepare features for match prediction"""
        
        # Get historical match data with team statistics
        query = """
        WITH team_form AS (
            SELECT 
                m.id as match_id,
                m.home_team_id,
                m.away_team_id,
                m.match_date,
                -- Home team recent form
                (SELECT AVG(CASE 
                    WHEN (m2.home_team_id = m.home_team_id AND m2.home_score > m2.away_score)
                    OR (m2.away_team_id = m.home_team_id AND m2.away_score > m2.home_score)
                    THEN 1 ELSE 0 END)
                 FROM matches m2 
                 WHERE (m2.home_team_id = m.home_team_id OR m2.away_team_id = m.home_team_id)
                   AND m2.match_date < m.match_date 
                   AND m2.status = 'finished'
                 ORDER BY m2.match_date DESC LIMIT %s) as home_form,
                
                -- Away team recent form
                (SELECT AVG(CASE 
                    WHEN (m2.home_team_id = m.away_team_id AND m2.home_score > m2.away_score)
                    OR (m2.away_team_id = m.away_team_id AND m2.away_score > m2.home_score)
                    THEN 1 ELSE 0 END)
                 FROM matches m2 
                 WHERE (m2.home_team_id = m.away_team_id OR m2.away_team_id = m.away_team_id)
                   AND m2.match_date < m.match_date 
                   AND m2.status = 'finished'
                 ORDER BY m2.match_date DESC LIMIT %s) as away_form,
                
                -- Home team recent goals for
                (SELECT AVG(CASE WHEN m2.home_team_id = m.home_team_id 
                    THEN m2.home_score ELSE m2.away_score END)
                 FROM matches m2 
                 WHERE (m2.home_team_id = m.home_team_id OR m2.away_team_id = m.home_team_id)
                   AND m2.match_date < m.match_date 
                   AND m2.status = 'finished'
                 ORDER BY m2.match_date DESC LIMIT %s) as home_avg_goals_for,
                
                -- Away team recent goals for
                (SELECT AVG(CASE WHEN m2.home_team_id = m.away_team_id 
                    THEN m2.home_score ELSE m2.away_score END)
                 FROM matches m2 
                 WHERE (m2.home_team_id = m.away_team_id OR m2.away_team_id = m.away_team_id)
                   AND m2.match_date < m.match_date 
                   AND m2.status = 'finished'
                 ORDER BY m2.match_date DESC LIMIT %s) as away_avg_goals_for,
                
                -- Head to head record
                (SELECT AVG(CASE 
                    WHEN m2.home_score > m2.away_score THEN 1
                    WHEN m2.home_score = m2.away_score THEN 0.5
                    ELSE 0 END)
                 FROM matches m2 
                 WHERE ((m2.home_team_id = m.home_team_id AND m2.away_team_id = m.away_team_id)
                    OR (m2.home_team_id = m.away_team_id AND m2.away_team_id = m.home_team_id))
                   AND m2.match_date < m.match_date 
                   AND m2.status = 'finished'
                 ORDER BY m2.match_date DESC LIMIT 10) as h2h_home_advantage,
                
                -- Match outcome
                CASE 
                    WHEN m.home_score > m.away_score THEN 'Home Win'
                    WHEN m.home_score = m.away_score THEN 'Draw'
                    ELSE 'Away Win'
                END as outcome,
                
                m.home_score,
                m.away_score
                
            FROM matches m
            WHERE m.status = 'finished'
              AND m.season_id = (SELECT id FROM seasons WHERE is_current = TRUE)
        )
        SELECT * FROM team_form 
        WHERE home_form IS NOT NULL 
          AND away_form IS NOT NULL
        ORDER BY match_date
        """
        
        params = [lookback_games] * 4
        df = self.db.execute_query(query, params)
        
        if df.empty:
            logger.warning("No data available for feature preparation")
            return pd.DataFrame()
        
        # Add additional features
        df['goal_difference_tendency'] = df['home_avg_goals_for'] - df['away_avg_goals_for']
        df['form_difference'] = df['home_form'] - df['away_form']
        df['is_manchester_united'] = ((df['home_team_id'] == self._get_mufc_id()) | 
                                     (df['away_team_id'] == self._get_mufc_id())).astype(int)
        
        # Handle missing values
        df = df.fillna(0.5)  # Fill with neutral values
        
        return df
    
    def train_model(self, test_size: float = 0.2) -> Dict:
        """Train the match prediction model"""
        
        df = self.prepare_features()
        
        if df.empty or len(df) < 20:
            logger.error("Insufficient data for model training")
            return {'error': 'Insufficient training data'}
        
        # Prepare features and target
        feature_cols = ['home_form', 'away_form', 'home_avg_goals_for', 'away_avg_goals_for',
                       'h2h_home_advantage', 'goal_difference_tendency', 'form_difference', 
                       'is_manchester_united']
        
        X = df[feature_cols].fillna(0)
        y = df['outcome']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.feature_columns = feature_cols
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
        
        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logger.info(f"Model trained with accuracy: {accuracy:.3f}")
        return results
    
    def predict_next_match(self, opponent_team_id: int, is_home: bool = True) -> Dict:
        """Predict outcome of next Manchester United match"""
        
        if self.model is None:
            logger.error("Model not trained. Call train_model() first.")
            return {'error': 'Model not trained'}
        
        mufc_id = self._get_mufc_id()
        
        # Prepare features for prediction
        if is_home:
            home_team_id, away_team_id = mufc_id, opponent_team_id
        else:
            home_team_id, away_team_id = opponent_team_id, mufc_id
        
        # Get recent form and statistics
        features = self._get_prediction_features(home_team_id, away_team_id)
        
        if not features:
            return {'error': 'Insufficient data for prediction'}
        
        # Make prediction
        feature_array = np.array([[
            features['home_form'],
            features['away_form'], 
            features['home_avg_goals_for'],
            features['away_avg_goals_for'],
            features['h2h_home_advantage'],
            features['goal_difference_tendency'],
            features['form_difference'],
            1  # is_manchester_united
        ]])
        
        feature_array_scaled = self.scaler.transform(feature_array)
        
        prediction = self.model.predict(feature_array_scaled)[0]
        probabilities = self.model.predict_proba(feature_array_scaled)[0]
        
        # Get class labels
        classes = self.model.classes_
        prob_dict = dict(zip(classes, probabilities))
        
        return {
            'prediction': prediction,
            'probabilities': prob_dict,
            'confidence': max(probabilities),
            'features_used': features,
            'is_home_match': is_home
        }
    
    def _get_mufc_id(self) -> int:
        """Get Manchester United team ID"""
        query = "SELECT id FROM teams WHERE short_name = 'MUN'"
        result = self.db.execute_query(query, fetch=True)
        return result[0]['id'] if result else 1
    
    def _get_prediction_features(self, home_team_id: int, away_team_id: int) -> Dict:
        """Get features for prediction"""
        
        query = """
        SELECT 
            -- Home team recent form (last 5 games)
            (SELECT AVG(CASE 
                WHEN (m.home_team_id = %s AND m.home_score > m.away_score)
                OR (m.away_team_id = %s AND m.away_score > m.home_score)
                THEN 1 ELSE 0 END)
             FROM matches m 
             WHERE (m.home_team_id = %s OR m.away_team_id = %s)
               AND m.status = 'finished'
             ORDER BY m.match_date DESC LIMIT 5) as home_form,
            
            -- Away team recent form
            (SELECT AVG(CASE 
                WHEN (m.home_team_id = %s AND m.home_score > m.away_score)
                OR (m.away_team_id = %s AND m.away_score > m.home_score)
                THEN 1 ELSE 0 END)
             FROM matches m 
             WHERE (m.home_team_id = %s OR m.away_team_id = %s)
               AND m.status = 'finished'
             ORDER BY m.match_date DESC LIMIT 5) as away_form,
             
            -- Goals for averages
            (SELECT AVG(CASE WHEN m.home_team_id = %s THEN m.home_score ELSE m.away_score END)
             FROM matches m 
             WHERE (m.home_team_id = %s OR m.away_team_id = %s)
               AND m.status = 'finished'
             ORDER BY m.match_date DESC LIMIT 5) as home_avg_goals_for,
             
            (SELECT AVG(CASE WHEN m.home_team_id = %s THEN m.home_score ELSE m.away_score END)
             FROM matches m 
             WHERE (m.home_team_id = %s OR m.away_team_id = %s)
               AND m.status = 'finished'
             ORDER BY m.match_date DESC LIMIT 5) as away_avg_goals_for,
             
            -- Head to head
            (SELECT AVG(CASE 
                WHEN m.home_score > m.away_score THEN 1
                WHEN m.home_score = m.away_score THEN 0.5
                ELSE 0 END)
             FROM matches m 
             WHERE ((m.home_team_id = %s AND m.away_team_id = %s)
                OR (m.home_team_id = %s AND m.away_team_id = %s))
               AND m.status = 'finished'
             ORDER BY m.match_date DESC LIMIT 10) as h2h_home_advantage
        """
        
        params = [home_team_id, home_team_id, home_team_id, home_team_id,
                 away_team_id, away_team_id, away_team_id, away_team_id,
                 home_team_id, home_team_id, home_team_id,
                 away_team_id, away_team_id, away_team_id,
                 home_team_id, away_team_id, away_team_id, home_team_id]
        
        result = self.db.execute_query(query, params)
        
        if result.empty:
            return {}
        
        row = result.iloc[0]
        features = {
            'home_form': row['home_form'] or 0.5,
            'away_form': row['away_form'] or 0.5,
            'home_avg_goals_for': row['home_avg_goals_for'] or 1.0,
            'away_avg_goals_for': row['away_avg_goals_for'] or 1.0,
            'h2h_home_advantage': row['h2h_home_advantage'] or 0.5
        }
        
        features['goal_difference_tendency'] = features['home_avg_goals_for'] - features['away_avg_goals_for']
        features['form_difference'] = features['home_form'] - features['away_form']
        
        return features
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

class PlayerPerformancePredictor:
    """Predict individual player performance"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.models = {}
        self.scalers = {}
    
    def prepare_player_features(self, player_id: int, lookback_games: int = 10) -> pd.DataFrame:
        """Prepare features for player performance prediction"""
        
        query = """
        SELECT 
            pms.*,
            m.match_date,
            -- Recent form features
            AVG(pms2.goals) OVER (
                PARTITION BY pms.player_id 
                ORDER BY m.match_date 
                ROWS BETWEEN %s PRECEDING AND 1 PRECEDING
            ) as recent_avg_goals,
            
            AVG(pms2.assists) OVER (
                PARTITION BY pms.player_id 
                ORDER BY m.match_date 
                ROWS BETWEEN %s PRECEDING AND 1 PRECEDING
            ) as recent_avg_assists,
            
            AVG(pms2.rating) OVER (
                PARTITION BY pms.player_id 
                ORDER BY m.match_date 
                ROWS BETWEEN %s PRECEDING AND 1 PRECEDING
            ) as recent_avg_rating,
            
            -- Opposition strength
            CASE 
                WHEN opp.final_position <= 6 THEN 'Strong'
                WHEN opp.final_position <= 14 THEN 'Medium'
                ELSE 'Weak'
            END as opposition_strength,
            
            -- Home/Away
            CASE WHEN m.home_team_id = pms.team_id THEN 1 ELSE 0 END as is_home
            
        FROM player_match_stats pms
        JOIN players p ON pms.player_id = p.id
        JOIN matches m ON pms.match_id = m.id
        LEFT JOIN player_match_stats pms2 ON pms2.player_id = pms.player_id
        LEFT JOIN opposition_categories opp ON opp.team_id = 
            CASE WHEN m.home_team_id = pms.team_id THEN m.away_team_id ELSE m.home_team_id END
        WHERE pms.player_id = %s
          AND pms.minutes_played > 0
          AND m.status = 'finished'
        ORDER BY m.match_date
        """
        
        params = [lookback_games, lookback_games, lookback_games, player_id]
        df = self.db.execute_query(query, params)
        
        # Encode categorical variables
        if not df.empty:
            df['opposition_strength'] = df['opposition_strength'].map({
                'Strong': 2, 'Medium': 1, 'Weak': 0
            }).fillna(1)
        
        return df.fillna(0)
    
    def train_player_model(self, player_id: int, target_metric: str = 'rating') -> Dict:
        """Train model for specific player and metric"""
        
        df = self.prepare_player_features(player_id)
        
        if df.empty or len(df) < 10:
            return {'error': 'Insufficient data for player model'}
        
        # Feature selection
        feature_cols = ['recent_avg_goals', 'recent_avg_assists', 'recent_avg_rating',
                       'opposition_strength', 'is_home', 'minutes_played']
        
        X = df[feature_cols].fillna(0)
        y = df[target_metric]
        
        # Remove rows where target is missing
        mask = ~y.isna()
        X, y = X[mask], y[mask]
        
        if len(X) < 5:
            return {'error': 'Insufficient valid data'}
        
        # Split data
        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        
        # Store model
        model_key = f"{player_id}_{target_metric}"
        self.models[model_key] = model
        self.scalers[model_key] = scaler
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
            'training_samples': len(X_train),
            'predictions_sample': list(zip(y_test.tolist()[:5], y_pred[:5].tolist()))
        }
    
    def predict_player_performance(self, player_id: int, opposition_strength: str = 'Medium', 
                                 is_home: bool = True, target_metric: str = 'rating') -> Dict:
        """Predict player performance for next match"""
        
        model_key = f"{player_id}_{target_metric}"
        
        if model_key not in self.models:
            # Train model if not available
            train_result = self.train_player_model(player_id, target_metric)
            if 'error' in train_result:
                return train_result
        
        # Get recent performance
        recent_data = self.prepare_player_features(player_id, lookback_games=5)
        
        if recent_data.empty:
            return {'error': 'No recent performance data'}
        
        # Get most recent stats
        recent_stats = recent_data.iloc[-1]
        
        # Prepare features
        opp_strength_map = {'Strong': 2, 'Medium': 1, 'Weak': 0}
        features = np.array([[
            recent_stats['recent_avg_goals'],
            recent_stats['recent_avg_assists'],
            recent_stats['recent_avg_rating'],
            opp_strength_map.get(opposition_strength, 1),
            1 if is_home else 0,
            90  # Assuming full match
        ]])
        
        # Scale and predict
        model = self.models[model_key]
        scaler = self.scalers[model_key]
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        
        return {
            'predicted_value': prediction,
            'recent_average': recent_stats[f'recent_avg_{target_metric}'] if f'recent_avg_{target_metric}' in recent_stats else recent_stats[target_metric],
            'opposition_strength': opposition_strength,
            'is_home': is_home,
            'confidence': 'Medium'  # Could be improved with prediction intervals
        }

# Integration with main system
class MLAnalyticsEngine:
    """Main ML engine integrating all models"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.match_predictor = MatchPredictor(db_manager)
        self.player_predictor = PlayerPerformancePredictor(db_manager)
    
    def train_all_models(self) -> Dict:
        """Train all available models"""
        results = {}
        
        # Train match prediction model
        logger.info("Training match prediction model...")
        match_results = self.match_predictor.train_model()
        results['match_prediction'] = match_results
        
        # Train player models for key players
        key_players = self._get_key_players()
        results['player_models'] = {}
        
        for player_id, player_name in key_players:
            logger.info(f"Training models for {player_name}...")
            
            for metric in ['rating', 'goals', 'assists']:
                player_results = self.player_predictor.train_player_model(player_id, metric)
                results['player_models'][f"{player_name}_{metric}"] = player_results
        
        return results
    
    def get_match_predictions(self, upcoming_fixtures: List[Dict]) -> List[Dict]:
        """Get predictions for upcoming matches"""
        predictions = []
        
        for fixture in upcoming_fixtures:
            try:
                prediction = self.match_predictor.predict_next_match(
                    fixture['opponent_id'],
                    fixture['is_home']
                )
                prediction['fixture'] = fixture
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Prediction failed for fixture {fixture}: {e}")
                continue
        
        return predictions
    
    def _get_key_players(self) -> List[Tuple[int, str]]:
        """Get key Manchester United players for modeling"""
        query = """
        SELECT p.id, p.name
        FROM players p
        JOIN player_match_stats pms ON p.id = pms.player_id
        WHERE p.team_id = (SELECT id FROM teams WHERE short_name = 'MUN')
          AND p.is_active = TRUE
        GROUP BY p.id, p.name
        HAVING COUNT(pms.match_id) >= 5
        ORDER BY SUM(pms.minutes_played) DESC
        LIMIT 15
        """
        
        result = self.db.execute_query(query)
        return [(row['id'], row['name']) for _, row in result.iterrows()]

# Usage example for Cloud Functions
def run_ml_training(request=None):
    """Cloud Function to train ML models"""
    from data_collector import DatabaseManager
    
    try:
        db_manager = DatabaseManager()
        ml_engine = MLAnalyticsEngine(db_manager)
        
        results = ml_engine.train_all_models()
        
        return {
            'status': 'success',
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"ML training failed: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Local testing
    from data_collector import DatabaseManager
    
    # Test configuration
    db_config = {
        'host': 'localhost',
        'database': 'mufc_analytics',
        'user': 'postgres',
        'password': 'password'
    }
    
    db_manager = DatabaseManager()
    
    # Test match predictor
    predictor = MatchPredictor(db_manager)
    training_results = predictor.train_model()
    print("Match Prediction Model Results:")
    print(f"Accuracy: {training_results.get('accuracy', 'N/A')}")
    print(f"Feature Importance: {training_results.get('feature_importance', {})}")