import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load dataset
df = pd.read_csv('health_data.csv', sep=',')

print("✅ Dataset Loaded Successfully!")
print(f"Dataset shape: {df.shape}")
print("\nFeature summary:")
print(df[['Heart Rate', 'Body Temperature', 'Oxygen Saturation', 'Age']].describe())

# Select features
selected_features = ['Heart Rate', 'Body Temperature', 'Oxygen Saturation', 'Age', 'Gender']
df = df[selected_features + ['Risk Category']]

# ===== ENHANCED RISK CLASSIFICATION =====
def classify_risk_detailed(row):
    """
    Multi-level risk classification based on medical thresholds
    Returns: Normal, Caution, Moderate Risk, High Risk, Critical
    """
    hr = row['Heart Rate']
    temp = row['Body Temperature']
    spo2 = row['Oxygen Saturation']
    age = row['Age']
    
    risk_score = 0
    
    # Heart Rate scoring
    if age < 18:
        if hr < 60 or hr > 100: risk_score += 2
        elif hr < 70 or hr > 90: risk_score += 1
    elif age < 65:
        if hr < 60 or hr > 100: risk_score += 2
        elif hr < 65 or hr > 95: risk_score += 1
    else:  # Elderly
        if hr < 55 or hr > 100: risk_score += 3
        elif hr < 65 or hr > 90: risk_score += 1.5
    
    # Body Temperature scoring
    if temp < 35.5 or temp > 38.5: risk_score += 3
    elif temp < 36.0 or temp > 37.8: risk_score += 2
    elif temp < 36.3 or temp > 37.5: risk_score += 1
    
    # SpO2 scoring (most critical)
    if spo2 < 90: risk_score += 5  # Critical
    elif spo2 < 94: risk_score += 3  # High risk
    elif spo2 < 96: risk_score += 1.5  # Moderate
    elif spo2 < 97: risk_score += 0.5  # Caution
    
    # Age factor
    if age > 70: risk_score += 1
    elif age > 60: risk_score += 0.5
    
    # Classification based on total score
    if risk_score >= 7:
        return 'Critical'
    elif risk_score >= 5:
        return 'High Risk'
    elif risk_score >= 3:
        return 'Moderate Risk'
    elif risk_score >= 1:
        return 'Caution'
    else:
        return 'Normal'

# Apply enhanced classification
df['Risk Category'] = df.apply(classify_risk_detailed, axis=1)

print("\n✅ Enhanced Risk Categories Applied!")
print("\nRisk Distribution:")
print(df['Risk Category'].value_counts())
print(f"\nPercentages:")
print(df['Risk Category'].value_counts(normalize=True) * 100)

# Encode Gender
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

# Encode Risk Category (5 levels)
le_risk = LabelEncoder()
df['Risk Category'] = le_risk.fit_transform(df['Risk Category'])

print(f"\nRisk Level Mapping:")
for i, label in enumerate(le_risk.classes_):
    print(f"  {i} = {label}")

# Split features and target
X = df[selected_features]
y = df['Risk Category']

# Standardize features
scaler = StandardScaler()
numeric_features = ['Heart Rate', 'Body Temperature', 'Oxygen Saturation', 'Age']
X_scaled = X.copy()
X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])

print("\n✅ Feature Scaling Complete!")

# Split dataset with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.25, 
    random_state=42, 
    stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ===== IMPROVED MODEL WITH BETTER PARAMETERS =====
model = RandomForestClassifier(
    n_estimators=200,           # More trees
    max_depth=15,               # Deeper trees for complex patterns
    min_samples_split=10,       # Prevent overfitting
    min_samples_leaf=5,         # Ensure leaf nodes have enough samples
    class_weight='balanced',    # Handle class imbalance
    random_state=42,
    n_jobs=-1                   # Use all CPU cores
)

print("\n🔄 Training Model...")
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"\n✅ Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Training Complete!")
print(f"Test Accuracy: {accuracy*100:.2f}%")

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=le_risk.classes_))

print("\nCONFUSION MATRIX")
print("="*60)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
importances = pd.DataFrame({
    'feature': selected_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("FEATURE IMPORTANCES")
print("="*60)
print(importances.to_string(index=False))

# Save all artifacts
print("\n💾 Saving model and artifacts...")
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(le_gender, open('gender_encoder.pkl', 'wb'))
pickle.dump(le_risk, open('risk_encoder.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(X[numeric_features].mean(), open('col_means.pkl', 'wb'))

# Save risk level mapping for reference
risk_mapping = {i: label for i, label in enumerate(le_risk.classes_)}
pickle.dump(risk_mapping, open('risk_mapping.pkl', 'wb'))

print("\n✅ All files saved successfully!")
print("\nFiles created:")
print("  - model.pkl (Random Forest model)")
print("  - gender_encoder.pkl (Gender encoder)")
print("  - risk_encoder.pkl (Risk category encoder)")
print("  - scaler.pkl (Feature scaler)")
print("  - col_means.pkl (Column means)")
print("  - risk_mapping.pkl (Risk level mapping)")

# ===== TESTING WITH SAMPLE DATA =====
print("\n" + "="*60)
print("TESTING MODEL WITH SAMPLE CASES")
print("="*60)

test_cases = [
    {'Heart Rate': 72, 'Body Temperature': 36.8, 'Oxygen Saturation': 98, 'Age': 30, 'Gender': 1, 'Expected': 'Normal'},
    {'Heart Rate': 95, 'Body Temperature': 37.5, 'Oxygen Saturation': 96, 'Age': 45, 'Gender': 0, 'Expected': 'Caution'},
    {'Heart Rate': 120, 'Body Temperature': 38.5, 'Oxygen Saturation': 88, 'Age': 70, 'Gender': 0, 'Expected': 'High Risk'},
]

for i, case in enumerate(test_cases, 1):
    expected = case.pop('Expected')
    test_df = pd.DataFrame([case])
    test_df[numeric_features] = scaler.transform(test_df[numeric_features])
    prediction = model.predict(test_df)[0]
    risk_level = le_risk.inverse_transform([prediction])[0]
    
    print(f"\nCase {i}: {expected}")
    print(f"  Input: HR={case['Heart Rate']}, Temp={case['Body Temperature']}, SpO2={case['Oxygen Saturation']}, Age={case['Age']}")
    print(f"  Predicted: {risk_level}")

print("\n" + "="*60)
print("✅ Model training and testing complete!")
print("="*60)