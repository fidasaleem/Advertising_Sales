import streamlit as st
import pandas as pd
import base64
import cloudpickle

# ------------------ Page Config ------------------
st.set_page_config(page_title="Advertising Sales Prediction App", page_icon="📈", layout="wide")

# ------------------ Background Image ------------------
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

set_bg("Background.jpg")  # Replace with your image

# ------------------ App Title ------------------
st.title("📊 Advertising Sales Prediction App")
st.markdown("""
This app predicts **product Sales** based on advertising expenditure across different media channels.  
Trained using a **Gradient Boosting Regressor (R² ≈ 0.96)** for high accuracy.
""")


def load_model():
    try:
        with open("gb_model.pkl", "rb") as f:
            model = cloudpickle.load(f)
        st.success("✅ Model loaded successfully!")
        st.write("Model type:", type(model))
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()


# ------------------ Sidebar Inputs ------------------
st.sidebar.header("🧮 Enter Advertising Spend")
tv = st.sidebar.number_input("💡 TV Advertising Budget ($)", min_value=0.0, step=1.0)
radio = st.sidebar.number_input("📻 Radio Advertising Budget ($)", min_value=0.0, step=1.0)
newspaper = st.sidebar.number_input("🗞️ Newspaper Advertising Budget ($)", min_value=0.0, step=1.0)

input_data = pd.DataFrame({
    "TV": [tv],
    "Radio": [radio],
    "Newspaper": [newspaper]
})

# ------------------ Prediction ------------------
if st.sidebar.button("🔮 Predict Sales"):
    if model is not None:
        try:
            # Make sure input has correct shape
            predicted_sales = model.predict(input_data)[0]
            st.subheader("🎯 Predicted Sales")
            st.success(f"Estimated Sales: **{predicted_sales:.2f} units**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Input DataFrame for debugging:")
            st.dataframe(input_data)
    else:
        st.warning("Model is not loaded. Cannot make predictions.")

# ------------------ Model Insights ------------------
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
