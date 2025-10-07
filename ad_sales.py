import streamlit as st
import pandas as pd
import base64
import cloudpickle


st.set_page_config(page_title="Advertising Sales Prediction App", page_icon="📈", layout="wide")

def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            height: 100vh;           
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage
set_bg("Background.jpg")  # Replace with your downloaded image name

# App Title
st.title("📊 Advertising Sales Prediction App")
st.markdown("""
This app predicts **product Sales** based on advertising expenditure across different media channels.  
Trained using a **Gradient Boosting Regressor (R² ≈ 0.96)** for high accuracy.
""")

# Load Pre-Trained Model
def load_model():
    with open("gb_model.pkl", "rb") as f:
        model = cloudpickle.load(f)
    return model
model = load_model()


# Sidebar Inputs
st.sidebar.header("🧮 Enter Advertising Spend")
tv = st.sidebar.number_input("💡 TV Advertising Budget ($)", min_value=0.0, step=1.0)
radio = st.sidebar.number_input("📻 Radio Advertising Budget ($)", min_value=0.0, step=1.0)
newspaper = st.sidebar.number_input("🗞️ Newspaper Advertising Budget ($)", min_value=0.0, step=1.0)

# Create input DataFrame
input_data = pd.DataFrame({
    "TV": [tv],
    "Radio": [radio],
    "Newspaper": [newspaper]
})

# Prediction
if st.sidebar.button("🔮 Predict Sales"):
    predicted_sales = model.predict(input_data)[0]

    st.subheader("🎯 Predicted Sales")
    st.success(f"Estimated Sales: **{predicted_sales:.2f} units**")

    # Business Insight Section
    st.markdown("---")
    st.markdown("### 🔍 Model Insights")
    st.markdown("""
    - **Gradient Boosting Regressor** provides highly accurate predictions.
    - **TV** and **Radio** are the strongest drivers of Sales.
    - **Newspaper** has a weaker but sometimes complementary effect.
    - Use this model to **optimize ad budgets** for maximum returns.
    """)
st.markdown("---")
st.caption("Developed by a Data Science Intern | Powered by Gradient Boosting in Python")