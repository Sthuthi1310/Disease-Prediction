import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

# Load dataset
df = pd.read_csv("data/original.csv")

# Clean data
df.columns = df.columns.str.strip()

for col in df.columns:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Get all symptoms
symptoms = set()

for col in df.columns[1:]:
    symptoms.update([str(x).strip().lower() for x in df[col].dropna().unique()])

symptoms = list(symptoms)

# Convert to binary format
new_data = []

for _, row in df.iterrows():
    row_dict = dict.fromkeys(symptoms, 0)

    for col in df.columns[1:]:
        symptom = str(row[col]).strip().lower()
        if symptom != "nan":
            row_dict[symptom] = 1

    row_dict["prognosis"] = row["Disease"]
    new_data.append(row_dict)

new_df = pd.DataFrame(new_data)

# Train-test split
train, test = train_test_split(new_df, test_size=0.2, random_state=42)

train.to_csv("data/Training.csv", index=False)
test.to_csv("data/Testing.csv", index=False)

# Train model
X_train = train.drop("prognosis", axis=1)
y_train = train["prognosis"]

model = GaussianNB()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(X_train.columns.tolist(), open("model/encoder.pkl", "wb"))

print("✅ Model trained successfully!")
