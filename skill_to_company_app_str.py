import streamlit as st
import pandas as pd
import joblib


model = joblib.load("skill_to_company_model.pkl")
encoder = joblib.load("skill_to_company_encoder.pkl")


menu = st.sidebar.selectbox("Menu", ["Home", "About", "Contact"])

st.title("🧠 Skill to Company Recommender")
user_name = st.text_input("Enter your name")
if menu == "Home":
    user_name = st.text_input("Welcome! Please enter your name to continue:")

    if user_name:
        st.write(f"Hello, {user_name}! Let's help you find the best company based on your skills.")

        with st.form("input_form"):
            skill = st.selectbox("Select your skill", ["Data Science", "Web Development", "UI/UX Design", "Cybersecurity", "AI"])
            location = st.selectbox("Your location", ["Kano", "Abuja", "Lagos", "Kaduna"])
            skill_level = st.selectbox("Your skill level", ["Beginner", "Intermediate", "Advanced"])
            work_type = st.selectbox("Preferred work type", ["Remote", "On-site", "Hybrid"])

            submitted = st.form_submit_button("Get Recommendation")

        if submitted:
            user_input = pd.DataFrame([{
                "skill": skill,
                "location": location,
                "skill_level": skill_level,
                "work_type": work_type
            }])

            transformed_input = encoder.transform(user_input)
            prediction = model.predict(transformed_input)

            st.success(f"✅ {user_name}, you are recommended to work with: {prediction[0]}")

elif menu == "About":
    st.subheader("About")
    st.write("This app helps recommend companies based on your digital skills and preferences.")

elif menu == "Contact":
    st.subheader("Contact")
    st.write("For inquiries, email us at: alameenahmadmunir@gmail.com")
   