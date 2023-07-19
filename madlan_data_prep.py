#!/usr/bin/env python
# coding: utf-8

# # matala madlan_data_prep

# ## MATAN & KARIN

# #### Import of required libraries:

# In[1]:


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from datetime import datetime


# #### prepare_data fanction:

# In[2]:


def prepare_data(data):
    
    data_copy = data.copy()
    # Delete assets where there is no price
    data_copy = data_copy[(data_copy['price'].notnull())]
    
    # Convert the price column to a numeric type 
    data_copy['price'] = data_copy['price'].str.replace('[^\d.]', '', regex=True)
    data_copy['price'] = data_copy['price'].str.extract('(\d+)', expand=False)
    data_copy['price'] = data_copy['price'].str.slice(stop=6)
    
    # Convert the Area column to a numeric type 
    data_copy['Area'] = data_copy['Area'].str.replace('[^\d.]', '', regex=True)
    data_copy['Area'] = data_copy['Area'].str.extract('(\d+)', expand=False)
    data_copy.loc[data_copy['Area'].str.len() > 3, 'Area'] = None
    data_copy['Area'] = pd.to_numeric(data_copy['Area'])
    
    # Remove unnecessary commas or punctuation from texts
    data_copy['Street'] = data_copy['Street'].str.replace(r'[^\w\s]', '', regex=False)
    data_copy['city_area'] = data_copy['city_area'].str.replace(r'[^\w\s]', '', regex=False)
    data_copy['description '] = data_copy['description '].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
    data_copy['publishedDays '] = data_copy['publishedDays '].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
    data_copy['entranceDate '] = data_copy['entranceDate '].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
    
    # Add a floor column 
    data_copy['floor '] = data_copy['floor_out_of'].str.split().str[1]
    data_copy['floor '] = data_copy['floor '].replace('קרקע','0')
    data_copy['floor '] = pd.to_numeric(data_copy['floor '], errors='coerce')
    
    # Add a total_floors column 
    data_copy['total_floors '] = data_copy['floor_out_of'].str.extract(r'מתוך\s(.*)')
    
    # Create a entrance_date column that is categorical
    import datetime

    data_copy['entranceDate '] = data_copy['entranceDate '].replace('גמיש', 'flexible')
    data_copy['entranceDate '] = data_copy['entranceDate '].replace('גמיש ', 'flexible')
    data_copy['entranceDate '] = data_copy['entranceDate '].replace('לא צויין', 'not_defined')
    data_copy['entranceDate '] = data_copy['entranceDate '].replace('מיידי', 'Less_than_6 months')

    def apply_date_conditions(date):
        # Convert the date object to a string
        date_str = date.strftime('%Y-%m-%d %H:%M:%S')

        # Convert the string back to a datetime object
        date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        today = datetime.now()
        # Calculate the time difference in months between the dates
        months_diff = (date_obj - today) // timedelta(days=30)

        if months_diff < 6:
            return 'less_than_6_months'
        elif 6 <= months_diff < 12:
            return 'months_6_12'
        else:
            return 'above_year'
    
    # Define the conditions and corresponding replacements
    conditions = {
      'גמיש': 'flexible',
      'גמיש ': 'flexible',
      'לא צויין': 'not_defined',
      'מיידי': 'less_than_6_months'}

    # Apply the conditions to the 'entranceDate' column
    data_copy['entrance_date'] = data_copy['entranceDate '].replace(conditions)

    # Identify date values and apply the date conditions
    date_mask = pd.to_datetime(data_copy['entrance_date'], errors='coerce').notna()
    data_copy.loc[date_mask, 'entrance_date'] = data_copy.loc[date_mask, 'entrance_date'].apply(apply_date_conditions)

    # Represents all Boolean fields as zeros and ones.
    #hasElevator column
    data_copy['hasElevator '] = data_copy['hasElevator '].replace(['יש מעלית', 'TRUE', 'yes'], 1)
    data_copy['hasElevator '] = data_copy['hasElevator '].replace(['אין מעלית', 'FALSE', 'no'], 0)
    #hasParking column
    data_copy['hasParking '] = data_copy['hasParking '].replace(['יש חנייה', 'TRUE', 'yes'], 1)
    data_copy['hasParking '] = data_copy['hasParking '].replace(['אין חנייה', 'FALSE', 'no'], 0)
    #hasStorage column
    data_copy['hasStorage '] = data_copy['hasStorage '].replace(['יש מחסן', 'TRUE', 'yes'], 1)
    data_copy['hasStorage '] = data_copy['hasStorage '].replace(['אין מחסן', 'FALSE', 'no'], 0)
    #hasAirCondition column
    data_copy['hasAirCondition '] = data_copy['hasAirCondition '].replace(['יש מיזוג אויר', 'TRUE', 'yes','כן','יש'], 1)
    data_copy['hasAirCondition '] = data_copy['hasAirCondition '].replace(['אין מיזוג אויר', 'FALSE', 'no','לא','אין'], 0)
    #hasBalcony column
    data_copy['hasBalcony '] = data_copy['hasBalcony '].replace(['יש מרפסת', 'TRUE', 'yes','כן','יש'], 1)
    data_copy['hasBalcony '] = data_copy['hasBalcony '].replace(['אין מרפסת', 'FALSE', 'no','לא','אין'], 0)
    #hasMamad column
    data_copy['hasMamad '] = data_copy['hasMamad '].replace(['יש ממ״ד', 'TRUE', 'yes','כן','יש'], 1)
    data_copy['hasMamad '] = data_copy['hasMamad '].replace([ 'אין ממ״ד', 'FALSE', 'no','לא','אין'], 0)
    #handicapFriendly column
    data_copy['handicapFriendly '] = data_copy['handicapFriendly '].replace([ 'נגיש לנכים', 'TRUE', 'נגיש'], 1)
    data_copy['handicapFriendly '] = data_copy['handicapFriendly '].replace([ 'לא נגיש לנכים', 'FALSE','לא נגיש'], 0)
    
    # Correction of city names
    data_copy["City"] = data_copy["City"].replace(" שוהם" , "שוהם")
    data_copy["City"] = data_copy["City"].replace("נהרייה" , "נהריה")
    data_copy["City"] = data_copy["City"].replace(" נהריה" , "נהריה")
    
    # Convert the room_number column to a numeric type 
    data_copy['room_number'] = data_copy['room_number'].str.replace('[^\d.]', '', regex=True)
    data_copy['room_number'] = data_copy['room_number'].str.extract('(\d+)', expand=False)
    data_copy = data_copy[(data_copy['room_number'].notnull())]
    data_copy.loc[:, 'room_number'] = data_copy['room_number'].astype(int)
    
    # Convert the Area column to a numeric type 
    def clean_area(value):
        if isinstance(value, int):
            return value
        elif isinstance(value, str):
            numbers = ''.join(filter(str.isdigit, value))
            if numbers:
                return int(numbers)
        return -1  # Return a default value when encountering NoneType
        
        data_copy = data_copy[(data_copy['Area'].notnull())]
        data_copy.loc[:, 'Area'] = data_copy['Area'].apply(clean_area)
        data_copy.loc[:, 'Area'] = data_copy['Area'].astype(int)
    
    # Creating data relevant to the model
    dataframe = pd.DataFrame(data_copy)
    
    return dataframe 
    
    


# In[3]:


#test:
datafile =  "C:\program_py\matalot\final_matala_py_and_machine_learnning\output_all_students_Train_v10.csv"
data = pd.read_csv(datafile)


# In[4]:


data1 = prepare_data(data)


# In[5]:


data1


# In[8]:


data1['entrance_date'].unique()


# In[ ]:





# In[ ]:




