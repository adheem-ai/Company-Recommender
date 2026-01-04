import pandas as pd
import joblib

model = joblib.load("company_model.pkl")
encoder = joblib.load("company_encoder.pkl")

user_input = pd.DataFrame({
   "skill": ['skill'],
   "location": ['location'],
   "skill_level": ['skill_level'],
   "work_type": ['work_type']       
})
user_input['skill'] = user_input['skill'].str.lower()
user_input['location'] = user_input['location'].str.lower()
user_input['skill_level'] = user_input['skill_level'].str.lower()
user_input['work_type'] = user_input['work_type'].str.lower()


user_input_enc = ct.transform(user_input)
recommendation = model.predict(user_input_enc)

print("Recommended Company:", recommendation[0])
