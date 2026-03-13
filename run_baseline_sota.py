"""
Medical AutoML - Comprehensive Baseline Comparison (5-Fold Cross Validation)

Compares SOTA deep learning models (TabNet, ResNet, MLP) 
with traditional ML baselines (Random Forest, XGBoost) for cardiovascular 
disease diagnosis using 5-Fold Cross Validation for fair comparison.

Usage: uv run python run_baseline_sota.py

All models use the same 5-fold splits (seed=42) for fair comparison with 
the Transformer model trained via train_kfold.py.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("Warning: TabNet not available. Install with: pip install pytorch-tabnet")
import warnings
import json
warnings.filterwarnings('ignore')

print("=" * 80)
print("Medical AutoML - SOTA Baseline Comparison (5-Fold Cross Validation)")
print("=" * 80)

# Configuration
RANDOM_STATE = 42
K_FOLDS = 5
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# =============================================================================
# 1. Data Loading
# =============================================================================
print("\n[1/3] Loading patient data...")

df = pd.read_csv('patients.csv')
df = df.replace('?', np.nan).dropna()

feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

X = df[feature_cols].astype(float)
y = df['num'].apply(lambda x: 1 if int(x) > 0 else 0)

print(f"  Dataset: {len(X)} patients, {len(feature_cols)} features")
print(f"  Class distribution: {dict(y.value_counts().sort_index())}")

# =============================================================================
# 2. Clinical Metrics Function
# =============================================================================
def calculate_clinical_metrics(y_true, y_pred, y_prob):
    """Calculate clinical-grade metrics: Acc, AUC, Sensitivity, Specificity."""
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': acc,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

# =============================================================================
# 3. Model Definitions (PyTorch)
# =============================================================================
class MLP(nn.Module):
    """Multi-Layer Perceptron."""
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
    """ResNet for tabular data."""
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

def train_pytorch_model(model_class, model_kwargs, X_train, y_train, X_val, y_val, device):
    """Train PyTorch model with early stopping."""
    model = model_class(**model_kwargs).to(device)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_auc = 0
    patience_counter = 0
    batch_size = 32
    
    for epoch in range(200):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        for i in range(0, X_train_t.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_t[indices], y_train_t[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_probs = F.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
            val_preds = val_outputs.argmax(dim=1).cpu().numpy()
            current_auc = roc_auc_score(y_val, val_probs)
        
        if current_auc > best_auc:
            best_auc = current_auc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        scheduler.step(1 - current_auc)
        
        if patience_counter >= 20:
            break
    
    # Load best model and return predictions
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_probs = F.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
        val_preds = val_outputs.argmax(dim=1).cpu().numpy()
    
    return val_preds, val_probs

# =============================================================================
# 4. 5-Fold Cross Validation Training
# =============================================================================
print("\n[2/3] Running 5-Fold Cross Validation...")

# Use StratifiedKFold with same seed as prepare_kfold.py
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# Store results for each model
all_results = {
    'Random Forest': [],
    'XGBoost': [],
    'Logistic Regression': [],
    'SVM (RBF)': [],
    'Gradient Boosting': [],
    'MLP (Deep)': [],
    'ResNet (Tabular)': [],
}

if TABNET_AVAILABLE:
    all_results['TabNet'] = []

# Collect predictions for ensemble
all_fold_predictions = {model: [] for model in all_results.keys()}

# Run CV
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*80}")
    print(f"Fold {fold_idx + 1}/{K_FOLDS}")
    print(f"{'='*80}")
    
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Standardize (fit on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 1. Random Forest
    print("  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_val)
    rf_probs = rf.predict_proba(X_val)[:, 1]
    all_results['Random Forest'].append(calculate_clinical_metrics(y_val, rf_preds, rf_probs))
    
    # 2. XGBoost
    print("  Training XGBoost...")
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                        use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_val)
    xgb_probs = xgb.predict_proba(X_val)[:, 1]
    all_results['XGBoost'].append(calculate_clinical_metrics(y_val, xgb_preds, xgb_probs))
    
    # 3. Logistic Regression
    print("  Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_val_scaled)
    lr_probs = lr.predict_proba(X_val_scaled)[:, 1]
    all_results['Logistic Regression'].append(calculate_clinical_metrics(y_val, lr_preds, lr_probs))
    
    # 4. SVM
    print("  Training SVM (RBF)...")
    svm = SVC(probability=True, kernel='rbf', random_state=RANDOM_STATE)
    svm.fit(X_train_scaled, y_train)
    svm_preds = svm.predict(X_val_scaled)
    svm_probs = svm.predict_proba(X_val_scaled)[:, 1]
    all_results['SVM (RBF)'].append(calculate_clinical_metrics(y_val, svm_preds, svm_probs))
    
    # 5. Gradient Boosting
    print("  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_val)
    gb_probs = gb.predict_proba(X_val)[:, 1]
    all_results['Gradient Boosting'].append(calculate_clinical_metrics(y_val, gb_preds, gb_probs))
    
    # 6. MLP
    print("  Training MLP...")
    input_dim = X_train.shape[1]
    mlp_preds, mlp_probs = train_pytorch_model(
        MLP, {'input_dim': input_dim, 'hidden_dims': [128, 64, 32], 'dropout': 0.3},
        X_train_scaled, y_train.values, X_val_scaled, y_val.values, device
    )
    all_results['MLP (Deep)'].append(calculate_clinical_metrics(y_val, mlp_preds, mlp_probs))
    
    # 7. ResNet
    print("  Training ResNet...")
    resnet_preds, resnet_probs = train_pytorch_model(
        ResNet, {'input_dim': input_dim, 'hidden_dim': 128, 'num_blocks': 3, 'dropout': 0.3},
        X_train_scaled, y_train.values, X_val_scaled, y_val.values, device
    )
    all_results['ResNet (Tabular)'].append(calculate_clinical_metrics(y_val, resnet_preds, resnet_probs))
    
    # 8. TabNet
    if TABNET_AVAILABLE:
        print("  Training TabNet...")
        try:
            tabnet_model = TabNetClassifier(
                n_d=64, n_a=64, n_steps=5, gamma=1.5, lambda_sparse=1e-4,
                optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
                mask_type='entmax', verbose=0
            )
            tabnet_model.fit(
                X_train=X_train_scaled, y_train=y_train.values,
                eval_set=[(X_val_scaled, y_val.values)],
                eval_name=['val'], eval_metric=['auc'],
                max_epochs=200, patience=20, batch_size=32,
                virtual_batch_size=16, num_workers=0, drop_last=False
            )
            tabnet_probs = tabnet_model.predict_proba(X_val_scaled)[:, 1]
            tabnet_preds = tabnet_model.predict(X_val_scaled)
            all_results['TabNet'].append(calculate_clinical_metrics(y_val, tabnet_preds, tabnet_probs))
        except Exception as e:
            print(f"    TabNet failed: {e}")
            all_results['TabNet'].append({'accuracy': 0, 'auc': 0, 'sensitivity': 0, 'specificity': 0})

# =============================================================================
# 5. Aggregate Results
# =============================================================================
print("\n" + "=" * 80)
print("5-FOLD CROSS VALIDATION RESULTS")
print("=" * 80)

# Calculate mean and std for each model
summary = {}
for model_name, fold_results in all_results.items():
    metrics = {}
    for metric in ['accuracy', 'auc', 'sensitivity', 'specificity']:
        values = [r[metric] for r in fold_results]
        metrics[metric] = {'mean': np.mean(values), 'std': np.std(values)}
    summary[model_name] = metrics

# Print individual fold results
print("\nIndividual Fold Results:")
print(f"{'Model':<25} {'Fold':<6} {'Accuracy':<12} {'AUC':<12} {'Sensitivity':<14} {'Specificity':<14}")
print("-" * 95)

for model_name in all_results.keys():
    for fold_idx, result in enumerate(all_results[model_name]):
        if fold_idx == 0:
            print(f"{model_name:<25} {fold_idx+1:<6} {result['accuracy']:<12.4f} "
                  f"{result['auc']:<12.4f} {result['sensitivity']:<14.4f} {result['specificity']:<14.4f}")
        else:
            print(f"{'':<25} {fold_idx+1:<6} {result['accuracy']:<12.4f} "
                  f"{result['auc']:<12.4f} {result['sensitivity']:<14.4f} {result['specificity']:<14.4f}")
    print("-" * 95)

# Print summary statistics
print("\n" + "=" * 80)
print("Summary Statistics (Mean ± Std)")
print("=" * 80)
print(f"{'Model':<25} {'Accuracy':<20} {'AUC':<20} {'Sensitivity':<20} {'Specificity':<20}")
print("-" * 105)

for model_name, metrics in summary.items():
    acc_str = f"{metrics['accuracy']['mean']:.4f} ± {metrics['accuracy']['std']:.4f}"
    auc_str = f"{metrics['auc']['mean']:.4f} ± {metrics['auc']['std']:.4f}"
    sens_str = f"{metrics['sensitivity']['mean']:.4f} ± {metrics['sensitivity']['std']:.4f}"
    spec_str = f"{metrics['specificity']['mean']:.4f} ± {metrics['specificity']['std']:.4f}"
    print(f"{model_name:<25} {acc_str:<20} {auc_str:<20} {sens_str:<20} {spec_str:<20}")

print("=" * 105)

# Rank by AUC
print("\n" + "=" * 80)
print("RANKING BY AUC (Mean ± Std)")
print("=" * 80)
ranked = sorted(summary.items(), key=lambda x: x[1]['auc']['mean'], reverse=True)
for i, (model_name, metrics) in enumerate(ranked, 1):
    auc_mean = metrics['auc']['mean']
    auc_std = metrics['auc']['std']
    print(f"{i}. {model_name:<25} AUC: {auc_mean:.4f} ± {auc_std:.4f}")

# Best model
best_model = ranked[0]
print(f"\n🏆 BEST BASELINE: {best_model[0]}")
print(f"   Accuracy:    {best_model[1]['accuracy']['mean']:.4f} ± {best_model[1]['accuracy']['std']:.4f}")
print(f"   AUC:         {best_model[1]['auc']['mean']:.4f} ± {best_model[1]['auc']['std']:.4f}")
print(f"   Sensitivity: {best_model[1]['sensitivity']['mean']:.4f} ± {best_model[1]['sensitivity']['std']:.4f}")
print(f"   Specificity: {best_model[1]['specificity']['mean']:.4f} ± {best_model[1]['specificity']['std']:.4f}")

# Comparison with Transformer
print("\n" + "=" * 80)
print("COMPARISON WITH YOUR TRANSFORMER MODEL")
print("=" * 80)
print("Transformer (from train_kfold.py): AUC = 0.910 ± 0.021")
best_auc_mean = best_model[1]['auc']['mean']
best_auc_std = best_model[1]['auc']['std']
print(f"Best Baseline ({best_model[0]}):     AUC = {best_auc_mean:.3f} ± {best_auc_std:.3f}")

# Statistical comparison
diff = 0.910 - best_auc_mean
if abs(diff) < 0.02:
    print(f"\n✅ Your Transformer is COMPARABLE to best baseline (diff: {diff:+.3f})")
elif diff > 0:
    print(f"\n🎉 Your Transformer OUTPERFORMS best baseline by {diff:.3f} AUC!")
else:
    print(f"\n⚠️  Your Transformer is {-diff:.3f} AUC below best baseline")

# Save results
results_dict = {
    'k_folds': K_FOLDS,
    'random_state': RANDOM_STATE,
    'individual_results': all_results,
    'summary': summary,
    'transformer_comparison': {
        'transformer_auc': '0.910 ± 0.021',
        'best_baseline': best_model[0],
        'best_baseline_auc': f"{best_auc_mean:.4f} ± {best_auc_std:.4f}"
    }
}

with open('baseline_comparison_5fold.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\n" + "=" * 80)
print("Results saved to: baseline_comparison_5fold.json")
print("=" * 80)
