# SkyCargo & Baggage Intelligence System

An executive flight operations dashboard built with Streamlit. The system integrates predictive machine learning models with linear payload optimization to evaluate baggage misconnection risks, maximize commercial cargo capacity utilization, and calculate financial delay liabilities in real time.

## Core Features
* Baggage Misconnect Risk Prediction: Machine learning inference engine predicting misconnection probability based on operational parameters (connecting windows, departure delays, passenger count). 
* Payload Capacity Optimization: Linear weight optimization engine allocating remaining aircraft payload capacity to high-margin commercial cargo.
* Financial Loss Liability Calculator: Dynamic delay penalty breakdown calculating passenger compensation, hotel accommodation, and food voucher liabilities.
* Executive Visualization: Custom dark-themed user interface featuring high-contrast operational analytics and weight distribution breakdowns.


## Project Structure

.
  app.py
  requirements.txt
  README.md
  src/
    data_pipeline.py
    model.py
    optimizer.py

## Local Installation & Setup

1. Clone the repository:
   git clone <>
   cd <>

2. Create and activate a virtual environment:
   python -m venv venv
   # On Windows (Command Prompt):
   venv\Scripts\activate
   
3. Install dependencies:
   pip install -r requirements.txt

4. Launch the dashboard:
   streamlit run app.py

## Deployment
This application is configured for single-click deployment on Streamlit Community Cloud:
1. Push your repository to GitHub.
2. Log into share.streamlit.io.
3. Connect your repository and select app.py as the main entry point.