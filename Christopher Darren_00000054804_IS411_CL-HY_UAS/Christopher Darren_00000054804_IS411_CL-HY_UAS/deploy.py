import numpy as np
import pickle
import streamlit as st

# run this in cmd:
# streamlit run I:\1.Project\Python\tensor\kuliah\damod\deploy.py

loaded_model = pickle.load(open('I:/1.Project/Python/tensor\kuliah\damod/best_xgbModel.pickle', 'rb'))
encoder = pickle.load(open('I:/1.Project/Python/tensor\kuliah\damod/encoder.pickle', 'rb'))
scaler = pickle.load(open('I:/1.Project/Python/tensor\kuliah\damod/scaler.pickle', 'rb'))
def tumor_prediction(input_data):

    cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col in cols:
        le = encoder[col]
        input_data[col] = le.transform([input_data[col]])[0]
    print(input_data)
    
    input_array = np.array(list(input_data.values()))
    input_array = input_array.astype(np.float64)
    input_array = scaler.transform([input_array])[0]
    prediction = loaded_model.predict([input_array])
    if (prediction[0] == 0):
        return 'Beresiko RENDAH stroke'
    else:
        return 'Beresiko TINGGI stroke'

def main():
    st.title('Stroke Prediction')

    gender = st.selectbox('Gender', options=['Male', 'Female', "Other"])
    age = st.number_input('Age', min_value=0)
    hypertension = st.number_input('Have hypertension? (No=0 | Yes=1)', min_value=0, max_value=1, step=1)
    heart_disease = st.number_input('Have heart disease? (No=0 | Yes=1)', min_value=0, max_value=1, step=1)
    ever_married = st.selectbox('Ever Married?', options=['Yes', 'No'])
    work_type = st.selectbox('Work Type', options=['Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked',])
    Residence_type = st.selectbox('Residence Type', options=['Urban', 'Rural'])
    avg_glucose_level = st.number_input('Average Glucose Level', min_value=0)
    bmi = st.number_input('Body Mass Index', min_value=0)
    smoking_status = st.selectbox('Smoking status', options=['never smoked', 'Unknown', 'formerly smoked', 'smokes'])
    diagnosis = ''
    
    if st.button('Tumor Test Result'):
        input_data = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": Residence_type,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status,
        }
        diagnosis = tumor_prediction(input_data)
        st.success(diagnosis)
    
if __name__ == '__main__':
    main()