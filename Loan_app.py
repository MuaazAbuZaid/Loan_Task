
import streamlit  as st
import joblib
import pandas as pd

Model = joblib.load("Model.pkl")
Inputs = joblib.load("Inputs.pkl")

def prediction(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):
    test_df = pd.DataFrame(columns=Inputs)
    test_df.at[0,"Gender"] = Gender
    test_df.at[0,"Married"] = Married
    test_df.at[0,"Dependents"] = Dependents
    test_df.at[0,"Education"] = Education
    test_df.at[0,"Self_Employed"] = Self_Employed
    test_df.at[0,"ApplicantIncome"] = ApplicantIncome
    test_df.at[0,"CoapplicantIncome"] = CoapplicantIncome
    test_df.at[0,"LoanAmount"] = LoanAmount
    test_df.at[0,"Loan_Amount_Term"] = Loan_Amount_Term
    test_df.at[0,"Credit_History"] = Credit_History
    test_df.at[0,"Property_Area"] = Property_Area
    result = Model.predict(test_df)
    return result[0]
def main():
    Gender = st.radio("Gender" ,['Male', 'Female'] )
    Married  = st.selectbox("Are you married ?" , ['Yes', 'No'])
    Dependents = st.selectbox("No. of Dependents: " , ['1', '2', '0', '3+'])
    Education = st.radio("I'm ..." ,['Graduate', 'Not Graduate'] )
    Self_Employed = st.selectbox("Are you Employed ?" , ['Yes', 'No'])
    Property_Area = st.selectbox("Property_Area" , ['Urban', 'Rural', 'Semiurban'])
    Credit_History = st.selectbox("Credit_History" , [1.0, 0.0])
    ApplicantIncome = st.slider("ApplicantIncome" , min_value= 0 , max_value=80000 , value=0,step=1000)
    CoapplicantIncome = st.slider("CoapplicantIncome" , min_value= 0 , max_value=80000 , value=0,step=1000)
    LoanAmount = st.slider("LoanAmount" , min_value= 0 , max_value=600 , value=0,step=1)
    Loan_Amount_Term = st.slider("Loan_Amount_Term_Years" , min_value= 0 , max_value=40 , value=0,step=1)
    if st.button("predict"):
        results = prediction(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area)
        status = ['Rejected', 'Approved']
        st.text(f"Loan is  {status[results]}")
if __name__ == '__main__':
    main()
