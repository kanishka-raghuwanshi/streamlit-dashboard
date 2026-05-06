import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from db import get_leads, get_customers

class LeadScoringModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = ['segment_num', 'region_num', 'source_num']
        self.is_trained = False
    
    def prepare_features(self, df):
        """Safe feature preparation - handles missing columns."""
        # Ensure required columns exist
        required_cols = ['segment', 'region', 'source']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 'SMB' if col == 'segment' else 'APAC' if col == 'region' else 'Organic'
        
        df_ml = df[['segment', 'region', 'source']].copy()
        
        for col in ['segment', 'region', 'source']:
            if col not in self.encoders:
                le = LabelEncoder()
                df_ml[f'{col}_num'] = le.fit_transform(df_ml[col].astype(str))
                self.encoders[col] = le
            else:
                df_ml[f'{col}_num'] = self.encoders[col].transform(df_ml[col].astype(str))
        
        return df_ml[self.feature_names]
    
    def generate_training_data(self, leads_df):
        X = self.prepare_features(leads_df)
        y = np.zeros(len(X))
        y[X['segment_num'] == 1] += 35  # Enterprise
        y[X['region_num'] == 2] += 25   # NA
        y[X['source_num'] == 2] += 20   # Referral
        y += np.random.normal(25, 12, len(y))
        y = np.clip(y, 0, 100)
        return X, y
    
    def train(self, leads_df):
        X, y = self.generate_training_data(leads_df)
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_r2 = r2_score(y_train, self.model.predict(X_train))
        test_r2 = r2_score(y_test, self.model.predict(X_test))
        print(f"✅ Lead Model: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}")
        return train_r2, test_r2
    
    def predict(self, leads_df):
        if not self.is_trained:
            self.train(leads_df)
        X = self.prepare_features(leads_df)
        X_scaled = self.scaler.transform(X)
        return np.clip(self.model.predict(X_scaled), 0, 100)

class ChurnPredictionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8)
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = ['segment_num', 'region_num', 'mrr_log', 'tenure_months']
        self.is_trained = False
    
    def prepare_features(self, df):
        """Safe feature preparation."""
        # Ensure required columns
        if 'segment' not in df.columns: df['segment'] = 'SMB'
        if 'region' not in df.columns: df['region'] = 'APAC'
        if 'mrr' not in df.columns: df['mrr'] = 5000
        if 'tenure_months' not in df.columns: df['tenure_months'] = 12
        
        df_ml = df[['segment', 'region', 'mrr', 'tenure_months']].copy()
        
        for col in ['segment', 'region']:
            if col not in self.encoders:
                le = LabelEncoder()
                df_ml[f'{col}_num'] = le.fit_transform(df_ml[col].astype(str))
                self.encoders[col] = le
            else:
                df_ml[f'{col}_num'] = self.encoders[col].transform(df_ml[col].astype(str))
        
        df_ml['mrr_log'] = np.log1p(df_ml['mrr'])
        return df_ml[self.feature_names]
    
    def generate_training_data(self, customers_df):
        X = self.prepare_features(customers_df)
        y = np.zeros(len(X))
        y[(X['segment_num'] == 0) & (X['mrr_log'] < 8)] = 1
        y[X['tenure_months'] < 6] += 0.3
        y = (np.random.rand(len(y)) < (0.4 + y * 0.3)).astype(int)
        return X, y
    
    def train(self, customers_df):
        X, y = self.generate_training_data(customers_df)
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        test_acc = accuracy_score(y_test, self.model.predict(X_test))
        print(f"✅ Churn Model: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")
        return train_acc, test_acc
    
    def predict_proba(self, customers_df):
        if not self.is_trained:
            self.train(customers_df)
        X = self.prepare_features(customers_df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

# Global instances
lead_model = LeadScoringModel()
churn_model = ChurnPredictionModel()

def train_all_models():
    """Train both models."""
    leads_df = get_leads()
    if len(leads_df) > 0:
        lead_model.train(leads_df)
    
    # Fixed: All arrays same length
    n_samples = 40
    customers_df = pd.DataFrame({
        'segment': np.random.choice(['Enterprise', 'SMB', 'Startup'], n_samples),
        'region': np.random.choice(['NA', 'APAC', 'EMEA', 'LATAM'], n_samples),
        'mrr': np.random.randint(1000, 50000, n_samples),
        'tenure_months': np.random.randint(1, 48, n_samples)
    })
    churn_model.train(customers_df)
    print("🤖 All ML Models Trained Successfully!")

if __name__ == "__main__":
    train_all_models()



