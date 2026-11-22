
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚦",
    layout="wide"
)

st.title("Road Accident Severity Prediction")
st.write(
    "Enter accident circumstances to obtain a predicted "
    "accident severity category."
)

# --------------------------------------------------
# LOAD TRAINED MODEL AND PREPROCESSOR
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(
        MODEL_DIR / "accident_severity_preprocessor.joblib"
    )

    model = joblib.load(
        MODEL_DIR / "accident_severity_classifier.joblib"
    )

    return preprocessor, model


try:
    preprocessor, model = load_artifacts()
except Exception as error:
    st.error(f"Could not load model artifacts: {error}")
    st.stop()

# --------------------------------------------------
# INPUT OPTIONS
# --------------------------------------------------

# Load the original dataset to obtain category options.
# This is used only to populate the input controls.
DATA_PATH = BASE_DIR / "RTA Dataset.csv"

@st.cache_data
def load_category_options():
    df = pd.read_csv(DATA_PATH)

    categorical_columns = [
        "Day_of_week",
        "Age_band_of_driver",
        "Sex_of_driver",
        "Educational_level",
        "Vehicle_driver_relation",
        "Driving_experience",
        "Type_of_vehicle",
        "Owner_of_vehicle",
        "Service_year_of_vehicle",
        "Defect_of_vehicle",
        "Area_accident_occured",
        "Lanes_or_Medians",
        "Road_allignment",
        "Types_of_Junction",
        "Road_surface_type",
        "Road_surface_conditions",
        "Light_conditions",
        "Weather_conditions",
        "Type_of_collision",
        "Vehicle_movement",
        "Pedestrian_movement",
        "Cause_of_accident"
    ]

    options = {}

    for column in categorical_columns:
        values = (
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        options[column] = sorted(values)

    return options


try:
    category_options = load_category_options()
except Exception as error:
    st.error(f"Could not load category options: {error}")
    st.stop()

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

st.subheader("Accident Information")

with st.form("accident_form"):

    col1, col2 = st.columns(2)

    with col1:
        accident_hour = st.slider(
            "Accident Hour",
            min_value=0,
            max_value=23,
            value=12
        )

        vehicles = st.number_input(
            "Number of Vehicles Involved",
            min_value=1,
            max_value=20,
            value=2
        )

        day = st.selectbox(
            "Day of Week",
            category_options["Day_of_week"]
        )

        age = st.selectbox(
            "Driver Age Band",
            category_options["Age_band_of_driver"]
        )

        sex = st.selectbox(
            "Driver Sex",
            category_options["Sex_of_driver"]
        )

        education = st.selectbox(
            "Educational Level",
            category_options["Educational_level"]
        )

        experience = st.selectbox(
            "Driving Experience",
            category_options["Driving_experience"]
        )

        driver_relation = st.selectbox(
            "Vehicle Driver Relation",
            category_options["Vehicle_driver_relation"]
        )

        vehicle_type = st.selectbox(
            "Vehicle Type",
            category_options["Type_of_vehicle"]
        )

        owner = st.selectbox(
            "Vehicle Owner",
            category_options["Owner_of_vehicle"]
        )

        service_year = st.selectbox(
            "Vehicle Service Year",
            category_options["Service_year_of_vehicle"]
        )

        defect = st.selectbox(
            "Vehicle Defect",
            category_options["Defect_of_vehicle"]
        )

    with col2:
        area = st.selectbox(
            "Accident Area",
            category_options["Area_accident_occured"]
        )

        lanes = st.selectbox(
            "Lanes or Medians",
            category_options["Lanes_or_Medians"]
        )

        alignment = st.selectbox(
            "Road Alignment",
            category_options["Road_allignment"]
        )

        junction = st.selectbox(
            "Types of Junction",
            category_options["Types_of_Junction"]
        )

        surface_type = st.selectbox(
            "Road Surface Type",
            category_options["Road_surface_type"]
        )

        surface_condition = st.selectbox(
            "Road Surface Condition",
            category_options["Road_surface_conditions"]
        )

        light = st.selectbox(
            "Light Conditions",
            category_options["Light_conditions"]
        )

        weather = st.selectbox(
            "Weather Conditions",
            category_options["Weather_conditions"]
        )

        collision = st.selectbox(
            "Type of Collision",
            category_options["Type_of_collision"]
        )

        movement = st.selectbox(
            "Vehicle Movement",
            category_options["Vehicle_movement"]
        )

        pedestrian = st.selectbox(
            "Pedestrian Movement",
            category_options["Pedestrian_movement"]
        )

        cause = st.selectbox(
            "Cause of Accident",
            category_options["Cause_of_accident"]
        )

    submitted = st.form_submit_button(
        "Predict Accident Severity",
        use_container_width=True
    )

# --------------------------------------------------
# TIME ENGINEERING
# --------------------------------------------------

def classify_time_of_day(hour):

    if 5 <= hour < 12:
        return "Morning"

    elif 12 <= hour < 17:
        return "Afternoon"

    elif 17 <= hour < 21:
        return "Evening"

    return "Night"


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    time_of_day = classify_time_of_day(accident_hour)

    input_data = pd.DataFrame([{
        "Number_of_vehicles_involved": vehicles,
        "Accident_hour": accident_hour,
        "Day_of_week": day,
        "Age_band_of_driver": age,
        "Sex_of_driver": sex,
        "Educational_level": education,
        "Vehicle_driver_relation": driver_relation,
        "Driving_experience": experience,
        "Type_of_vehicle": vehicle_type,
        "Owner_of_vehicle": owner,
        "Service_year_of_vehicle": service_year,
        "Defect_of_vehicle": defect,
        "Area_accident_occured": area,
        "Lanes_or_Medians": lanes,
        "Road_allignment": alignment,
        "Types_of_Junction": junction,
        "Road_surface_type": surface_type,
        "Road_surface_conditions": surface_condition,
        "Light_conditions": light,
        "Weather_conditions": weather,
        "Type_of_collision": collision,
        "Vehicle_movement": movement,
        "Pedestrian_movement": pedestrian,
        "Cause_of_accident": cause,
        "Time_of_day": time_of_day
    }])

    # Apply saved preprocessing
    transformed_input = preprocessor.transform(input_data)

    # Predict severity
    prediction = model.predict(transformed_input)[0]

    st.divider()

    st.subheader("Prediction Result")

    st.success(f"Predicted Accident Severity: {prediction}")

    # Display class probabilities if supported
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            transformed_input
        )[0]

        probability_df = pd.DataFrame({
            "Accident Severity": model.classes_,
            "Estimated Model Probability": probabilities
        })

        probability_df = probability_df.sort_values(
            by="Estimated Model Probability",
            ascending=False
        )

        st.subheader("Model Probability Estimates")

        st.bar_chart(
            probability_df.set_index("Accident Severity")
        )

        st.caption(
            "These values are model probability estimates, "
            "not calibrated real-world accident risks."
        )
