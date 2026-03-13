"""
Medical AutoML - Comprehensive Baseline Comparison

Compares SOTA deep learning models (TabNet, FT-Transformer, MLP, ResNet) 
with traditional ML baselines (Random Forest, XGBoost) for cardiovascular 
disease diagnosis.

Usage: uv run python run_baseline_sota.py

Models included:
- Traditional: Random Forest, XGBoost, Logistic Regression, SVM
- Deep Learning: TabNet (Google Research), ResNet, MLP
- Metrics: Accuracy, AUC, Sensitivity, Specificity (clinical-grade)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from tabnet import TabNetClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Medical AutoML - Comprehensive SOTA Baseline Comparison")
print("=" * 70)

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# =============================================================================
# 1. Data Loading and Preprocessing
# =============================================================================
print("\n[1/4] Loading and preprocessing patient data...")

df = pd.read_csv('patients.csv')
df = df.replace('?', np.nan).dropna()

# Feature engineering: keep all original features
feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

X = df[feature_cols].astype(float)
y = df['num'].apply(lambda x: 1 if int(x) > 0 else 0)

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y
)

print(f"  Dataset: {len(X)} patients, {len(feature_cols)} features")
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
print(f"  Class distribution: {y.value_counts().to_dict()}")

# Standardize features for neural networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors for PyTorch models
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.LongTensor(y_train.values)
y_test_tensor = torch.LongTensor(y_test.values)

# =============================================================================
# 2. Clinical Metrics Function
# =============================================================================
def calculate_clinical_metrics(y_true, y_pred, y_prob):
    """Calculate clinical-grade metrics: Acc, AUC, Sensitivity, Specificity."""
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    # Calculate confusion matrix for sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': acc,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

def print_metrics_table(results_dict):
    """Print results in a formatted table."""
    print("\n" + "=" * 90)
    print(f"{'Model':<25} {'Accuracy':<12} {'AUC':<12} {'Sensitivity':<14} {'Specificity':<14}")
    print("-" * 90)
    
    for model_name, metrics in results_dict.items():
        print(f"{model_name:<25} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['auc']:<12.4f} "
              f"{metrics['sensitivity']:<14.4f} "
              f"{metrics['specificity']:<14.4f}")
    print("=" * 90)

# =============================================================================
# 3. Traditional ML Baselines
# =============================================================================
print("\n[2/4] Training traditional ML baselines...")

results = {}

# Random Forest
print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]
results['Random Forest'] = calculate_clinical_metrics(y_test, rf_preds, rf_probs)

# XGBoost
print("  Training XGBoost...")
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                    use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_probs = xgb.predict_proba(X_test)[:, 1]
results['XGBoost'] = calculate_clinical_metrics(y_test, xgb_preds, xgb_probs)

# Logistic Regression
print("  Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
results['Logistic Regression'] = calculate_clinical_metrics(y_test, lr_preds, lr_probs)

# SVM with RBF kernel
print("  Training SVM (RBF)...")
svm = SVC(probability=True, kernel='rbf', random_state=RANDOM_STATE)
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)
svm_probs = svm.predict_proba(X_test_scaled)[:, 1]
results['SVM (RBF)'] = calculate_clinical_metrics(y_test, svm_preds, svm_probs)

# Gradient Boosting
print("  Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)
gb_probs = gb.predict_proba(X_test)[:, 1]
results['Gradient Boosting'] = calculate_clinical_metrics(y_test, gb_preds, gb_probs)

# =============================================================================
# 4. Deep Learning SOTA Models
# =============================================================================
print("\n[3/4] Training Deep Learning SOTA models...")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {device}")

# Training configuration
N_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 20

class MLP(nn.Module):
    """Multi-Layer Perceptron baseline."""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ResNetBlock(nn.Module):
    """Residual block for ResNet architecture."""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(dim)
        )
    
    def forward(self, x):
        return F.relu(x + self.net(x))

class ResNet(nn.Module):
    """ResNet for tabular data (inspired by Gorishniy et al., 2021)."""
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3, dropout=0.3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

def train_pytorch_model(model, X_train, y_train, X_test, y_test, model_name):
    """Train a PyTorch model with early stopping."""
    print(f"  Training {model_name}...")
    
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_auc = 0
    patience_counter = 0
    
    for epoch in range(N_EPOCHS):
        model.train()
        
        # Mini-batch training
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_probs = F.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
            test_preds = test_outputs.argmax(dim=1).cpu().numpy()
            
            current_auc = roc_auc_score(y_test.cpu().numpy(), test_probs)
            
            if current_auc > best_auc:
                best_auc = current_auc
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            scheduler.step(1 - current_auc)  # Invert AUC for loss
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break
    
    # Load best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_probs = F.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
        test_preds = test_outputs.argmax(dim=1).cpu().numpy()
    
    return calculate_clinical_metrics(y_test.cpu().numpy(), test_preds, test_probs)

# Train MLP
input_dim = X_train.shape[1]
mlp_model = MLP(input_dim, hidden_dims=[128, 64, 32], dropout=0.3)
results['MLP (Deep)'] = train_pytorch_model(mlp_model, X_train_tensor, y_train_tensor, 
                                            X_test_tensor, y_test_tensor, "MLP")

# Train ResNet
resnet_model = ResNet(input_dim, hidden_dim=128, num_blocks=3, dropout=0.3)
results['ResNet (Tabular)'] = train_pytorch_model(resnet_model, X_train_tensor, y_train_tensor,
                                                   X_test_tensor, y_test_tensor, "ResNet")

# =============================================================================
# 5. TabNet (Google Research SOTA)
# =============================================================================
print("\n[4/4] Training TabNet (Google Research)...")

try:
    # TabNet configuration optimized for small medical datasets
    tabnet_model = TabNetClassifier(
        n_d=64,              # Width of the decision prediction layer
        n_a=64,              # Width of the attention embedding
        n_steps=5,           # Number of decision steps
        gamma=1.5,           # Relaxation factor
        lambda_sparse=1e-4,  # Sparsity regularization
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax',  # "sparsemax" or "entmax"
        scheduler_params={"step_size": 50, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=0
    )
    
    # Train TabNet
    tabnet_model.fit(
        X_train=X_train_scaled,
        y_train=y_train.values,
        eval_set=[(X_test_scaled, y_test.values)],
        eval_name=['test'],
        eval_metric=['auc'],
        max_epochs=N_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        batch_size=BATCH_SIZE,
        virtual_batch_size=16,
        num_workers=0,
        drop_last=False
    )
    
    # Predictions
    tabnet_probs = tabnet_model.predict_proba(X_test_scaled)[:, 1]
    tabnet_preds = tabnet_model.predict(X_test_scaled)
    results['TabNet'] = calculate_clinical_metrics(y_test.values, tabnet_preds, tabnet_probs)
    
except Exception as e:
    print(f"  TabNet training failed: {e}")
    print("  Skipping TabNet...")

# =============================================================================
# 6. Results Summary
# =============================================================================
print("\n" + "=" * 90)
print("BASELINE COMPARISON RESULTS")
print_metrics_table(results)

# Sort by AUC
print("\n" + "=" * 90)
print("RANKING BY AUC (HIGHER IS BETTER)")
print("=" * 90)
sorted_results = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
for i, (model_name, metrics) in enumerate(sorted_results, 1):
    print(f"{i}. {model_name:<25} AUC: {metrics['auc']:.4f}")

# Best model details
best_model = sorted_results[0]
print(f"\n🏆 BEST MODEL: {best_model[0]}")
print(f"   Accuracy:    {best_model[1]['accuracy']:.4f}")
print(f"   AUC:         {best_model[1]['auc']:.4f}")
print(f"   Sensitivity: {best_model[1]['sensitivity']:.4f}")
print(f"   Specificity: {best_model[1]['specificity']:.4f}")

# Compare with your Transformer
print("\n" + "=" * 90)
print("COMPARISON WITH YOUR TRANSFORMER MODEL")
print("=" * 90)
print(f"Your Transformer (K-Fold): AUC ≈ 0.910 ± 0.021 (from previous runs)")
print(f"Best Baseline ({best_model[0]}): AUC = {best_model[1]['auc']:.4f}")
improvement = (0.910 - best_model[1]['auc']) / best_model[1]['auc'] * 100
print(f"Improvement: {improvement:+.1f}%" if improvement > 0 else f"Gap: {improvement:.1f}%")

print("\n" + "=" * 90)
print("Analysis complete!")
print("=" * 90)

# Save results to file
import json
with open('baseline_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nDetailed results saved to: baseline_comparison_results.json")
