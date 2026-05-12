import streamlit as st
from backend import determine_placement_posterior_probability

st.set_page_config(page_title="Placement Probability Predictor", layout="centered")
st.title("Placement Probability Predictor")
st.caption("Frontend:Streamlit | Backend:Rest API using FastAPI")
st.subheader("Please Enter your Details Below")

iq = st.number_input(label="Enter your IQ here", min_value=40, max_value=160, step=1)
previous_semester_result = st.number_input(label="Enter your Previous semester result here", min_value=0.0, max_value=10.0, step=0.01)
cgpa = st.number_input(label="Enter your CGPA here", min_value=0.0, max_value=10.0, step=0.01)
communication_skills = st.number_input("Enter your communication skills rating here", min_value=0, max_value=10, step=1)
projects_completed = st.number_input("Enter number of projects completed by you", min_value=0, max_value=5, step=1)

text_area_placeholder = st.empty()
if st.button("Calculate Probability of Getting Placed"):
    values = [iq, previous_semester_result, cgpa, communication_skills, projects_completed]
    response = determine_placement_posterior_probability(values)
    text_area_placeholder.text_area(label="Your Placement Probability Result", value=response["result"], height=200)
