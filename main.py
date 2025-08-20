import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import joblib
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    error = None
    form_data = {}

    if request.method == 'POST':
        try:
            # Load model and encoders
            model_data = joblib.load('sales_model.pkl')
            model = model_data['model']
            encoders = model_data['encoders']

            # Get form data
            form_data = {
                'Item_Identifier': request.form.get('Item_Identifier', ''),
                'Item_Weight': float(request.form.get('Item_Weight') or 0),
                'Item_Fat_Content': request.form.get('Item_Fat_Content', ''),
                'Item_Visibility': float(request.form.get('Item_Visibility') or 0),
                'Item_Type': request.form.get('Item_Type', ''),
                'Item_MRP': float(request.form.get('Item_MRP') or 0),
                'Outlet_Identifier': request.form.get('Outlet_Identifier', ''),
                'Outlet_Establishment_Year': int(request.form.get('Outlet_Establishment_Year') or 0),
                'Outlet_Size': request.form.get('Outlet_Size', ''),
                'Outlet_Location_Type': request.form.get('Outlet_Location_Type', ''),
                'Outlet_Type': request.form.get('Outlet_Type', '')
            }

            # Create DataFrame
            df = pd.DataFrame([form_data])

            # Encode categorical variables
            for col, encoder in encoders.items():
                if col in df.columns:
                    try:
                        df[col] = encoder.transform([str(df[col].iloc[0])])[0]
                    except:
                        df[col] = 0  # Default for unseen categories

            # Make prediction
            prediction = model.predict(df)[0]
            prediction = round(prediction, 2)

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction, error=error)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT
    app.run(host="0.0.0.0", port=port, debug=False)
