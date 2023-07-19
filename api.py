#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    # Get the data from the form
    data = request.form
    
    City = str(data['City'])
    typei = str(data['type'])
    room_number = int(data['room_number'])
    floor = int(data['floor '])
    Area = int(data['Area'])
    
    
    # Create a feature array
    data = {'City': City, 'type': typei, 'room_number': room_number,'floor ': floor, 'Area': Area}
    dataframe = pd.DataFrame(data, index=[0])
    
    # Predict
    y_pred =round(model.predict(dataframe)[0])

    # Return predicted price
    return render_template('index.html', price=y_pred)

if __name__ == '__main__':
    app.run()


# In[ ]:




