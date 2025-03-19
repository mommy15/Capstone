import streamlit as st
from ml_model import predict_rent_payment
from backend import register_user, login_user, add_rent_payment, get_rent_payments
import datetime

# ✅ Ensure `set_page_config()` is the FIRST Streamlit command
st.set_page_config(page_title="Rent Tracker", layout="centered")

# ✅ Inject Custom CSS for Gradient Background
def apply_custom_css():
    st.markdown("""
        <style>
        /* Full-page Gradient Background */
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #000000, #000000, #2D3B40, #52636A, #7F97A0);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 50% 100%; }
            100% { background-position: 100% 0%; }
        }

        /* Adjust Input Fields */
        input {
            border-radius: 8px !important;
            padding: 12px !important;
            border: 1px #0000 !important;
            
        }

        /* Custom Button Styling */
        .stButton>button {
            background-color: #FAFAFA !important;
            color: black !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 10px 15px !important;
            transition: 0.3s ease-in-out;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 20); 
        }
        .stButton>button:hover {
            background-color: #FAFAFA !important;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 20); 
        }
        </style>
    """, unsafe_allow_html=True)

# ✅ Apply the Custom CSS
apply_custom_css()

# ✅ Session State for Authentication
if "token" not in st.session_state:
    st.session_state["token"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None

# ✅ Use Tabs for Navigation Instead of Sidebar
tab1, tab2, tab3 = st.tabs(["Register", "Login", "Dashboard"])

# ✅ Register Tab
with tab1:
    st.subheader(" Register New User")
    email = st.text_input(" Email", key="reg_email")
    password = st.text_input(" Password", type="password", key="reg_password")
    role = st.selectbox(" Register as", ["tenant", "landlord"], key="reg_role")

    # Landlord email only required for tenant registration
    landlord_email = None
    if role == "tenant":
        landlord_email = st.text_input(" Landlord Email", key="reg_landlord_email")

    if st.button(" Register", key="register_btn"):
        response = register_user(email, password, role, landlord_email)
        st.success(response["message"] if "message" in response else response["error"])

        # ✅ Automatically switch to Login tab (Tab 2)
        if "message" in response:
            st.session_state["active_tab"] = "Login"


# ✅ Login Tab
with tab2:
    st.subheader(" User Login")
    email = st.text_input(" Email", key="login_email")
    password = st.text_input(" Password", type="password", key="login_password")
    
    if st.button(" Login", key="login_btn"):
        response = login_user(email, password)
        if "token" in response:
            st.session_state["token"] = response["token"]
            st.session_state["role"] = response["role"]
            st.success(" Logged in successfully!")
        else:
            st.error(response["error"])

# ✅ Dashboard Tab (Only Visible After Login)
if st.session_state["token"]:
    with tab3:
        st.title(" Rent Tracker Dashboard")

        # ✅ Add Rent Payment Form
        st.subheader(" Add Rent Payment")
        col1, col2 = st.columns(2)
        with col1:
            monthly_income = st.number_input(" Monthly Income (₹)", min_value=1000, step=500, key="add_income")
            amount = st.number_input(" Rent Amount (₹)", min_value=500, step=50, key="add_rent")
        
        with col2:
            payment_delay = st.number_input(" Payment Delay (days)", min_value=0, step=1, key="add_payment_delay")
            past_late_payments = st.number_input(" Previous Late Payments", min_value=0, step=1, key="add_past_late")

        # If landlord, specify tenant email
        if st.session_state["role"] == "landlord":
            tenant_email = st.text_input(" Tenant Email", key="add_tenant_email")
        else:
            tenant_email = email  # Tenant adds their own rent

        if st.button(" Add Payment", key="add_payment_btn"):
            response = add_rent_payment(tenant_email, email, monthly_income, amount, payment_delay, past_late_payments)
            st.success(response["message"])

        # ✅ Show Rent History
        st.subheader(" Rent Payment History")
        rent_data = get_rent_payments(email, st.session_state["role"])
        if isinstance(rent_data, list) and rent_data:
            st.table(rent_data)
        else:
            st.info("No rent payment history available.")

        # ✅ Rent Payment Prediction
        st.subheader(" Predict Future Rent Payment Behavior")
        col3, col4 = st.columns(2)
        with col3:
            predict_monthly_income = st.number_input(" Monthly Income (₹)", min_value=1000, step=500, key="predict_income")
            predict_rent_amount = st.number_input(" Rent Amount (₹)", min_value=500, step=50, key="predict_rent")
        
        with col4:
            predict_payment_delay = st.number_input(" Payment Delay (days)", min_value=0, step=1, key="predict_payment_delay")
            predict_past_late_payments = st.number_input(" Previous Late Payments", min_value=0, step=1, key="predict_past_late")

        # Predict Button
        if st.button(" Predict Rent Payment", key="predict_btn"):
            features = {
                "monthly_income": predict_monthly_income,
                "rent_amount": predict_rent_amount,
                "payment_delay": predict_payment_delay,
                "previous_late_payments": predict_past_late_payments
            }
            prediction = predict_rent_payment(features)
            st.success(f" Prediction: {prediction}")






