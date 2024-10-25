
!pip install fuzzywuzzy

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import fuzz, process # This import should work now
import numpy as np


# Load data
data = pd.read_csv('/content/amfi_nav_data.csv')

# Convert NAV to numeric
data['NAV'] = pd.to_numeric(data['NAV'], errors='coerce')
data = data.dropna(subset=['NAV'])

# Define risk level based on NAV
def risk_level(nav):
    if nav < 20:
        return 'Low Risk'
    elif 20 <= nav < 100:
        return 'Moderate Risk'
    else:
        return 'High Risk'

# Define term based on NAV
def investment_term(nav):
    if nav < 20:
        return 'Short-term'
    elif 20 <= nav < 100:
        return 'Medium-term'
    else:
        return 'Long-term'

# Apply labeling functions
data['Risk_Level'] = data['NAV'].apply(risk_level)
data['Term'] = data['NAV'].apply(investment_term)

# Define target variable based on moderate risk preference (Balanced Fund type)
data['Investment_Type'] = data.apply(
    lambda row: 'Balanced Fund' if row['Risk_Level'] == 'Moderate Risk' else 'Other', axis=1
)

# Encode the 'Category' column using embeddings
category_encoder = LabelEncoder()
data['Category_encoded'] = category_encoder.fit_transform(data['Category'])

# Define features and target
X = data[['Category_encoded', 'NAV']]
y = data['Investment_Type'].apply(lambda x: 1 if x == 'Balanced Fund' else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TensorFlow dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train.values, y_train)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((X_test.values, y_test)).batch(32)

# Build the model with embeddings
embedding_dim = 10
input_category = tf.keras.layers.Input(shape=(1,))
input_nav = tf.keras.layers.Input(shape=(1,))

# Embedding layer for category
category_embedding = tf.keras.layers.Embedding(
    input_dim=len(category_encoder.classes_),
    output_dim=embedding_dim,
    input_length=1
)(input_category)
category_embedding = tf.keras.layers.Flatten()(category_embedding)

# Concatenate embedding with NAV input
combined_input = tf.keras.layers.concatenate([category_embedding, input_nav])

# Dense layers
x = tf.keras.layers.Dense(64, activation='relu')(combined_input)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Define model
model = tf.keras.models.Model(inputs=[input_category, input_nav], outputs=output)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit([X_train['Category_encoded'], X_train['NAV']], y_train, epochs=30, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate([X_test['Category_encoded'], X_test['NAV']], y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Function to take user input and predict suitable fund types with typo tolerance
def predict_user_input():
    try:
        # Expected values for term and risk levels
        valid_terms = ['Short-term', 'Medium-term', 'Long-term']
        valid_risks = ['Low Risk', 'Moderate Risk', 'High Risk']

        # Take user input for term and risk level
        term_input = input("Enter the investment term (Short-term, Medium-term, Long-term): ")
        risk_level_input = input("Enter the risk level (Low Risk, Moderate Risk, High Risk): ")

        # Fuzzy match term and risk level inputs to handle typos
        term_input = process.extractOne(term_input, valid_terms, scorer=fuzz.ratio)[0]
        risk_level_input = process.extractOne(risk_level_input, valid_risks, scorer=fuzz.ratio)[0]

        # Filter data based on user input
        suitable_funds = data[(data['Risk_Level'] == risk_level_input) & (data['Term'] == term_input)]
        if suitable_funds.empty:
            print("No suitable funds found for your criteria.")
            return

        # Encode categories and NAV for prediction
        categories_encoded = category_encoder.transform(suitable_funds['Category'])
        nav_values = suitable_funds['NAV'].values

        # Predict suitability
        predictions = model.predict([categories_encoded, nav_values])
        suitable_funds['Prediction'] = predictions
        recommended_funds = suitable_funds[suitable_funds['Prediction'] >= 0.5]

        # Display recommended funds with Category, Name, and NAV
        if not recommended_funds.empty:
            print("Recommended funds for your criteria:")
            print(recommended_funds[['Category', 'Name', 'NAV', 'Risk_Level', 'Term']])
        else:
            print("No balanced funds found for your criteria.")
    except Exception as e:
        print("Error:", e)

# Prompt the user for input after model training and evaluation
predict_user_input()

import pickle
with open('fund_recommendation_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'category_encoder': category_encoder,
        'data': data
    }, f)

