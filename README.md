# SkyCargo & Baggage Intelligence System

An executive flight operations dashboard built with Streamlit. The system integrates predictive machine learning models with linear payload optimization to evaluate baggage misconnection risks, maximize commercial cargo capacity utilization, and calculate financial delay liabilities in real time.

## Core Features
* Baggage Misconnect Risk Prediction: Machine learning inference engine predicting misconnection probability based on operational parameters (connecting windows, departure delays, passenger count).
  <img width="422" height="356" alt="image" src="https://github.com/user-attachments/assets/0f37061e-c55a-475d-a025-363bed11174e" />

* Payload Capacity Optimization: Linear weight optimization engine allocating remaining aircraft payload capacity to high-margin commercial cargo.
  <img width="374" height="160" alt="image" src="https://github.com/user-attachments/assets/d18b0c3d-e4cf-4b0d-af7e-6d9cc5ad38d7" />

* Financial Loss Liability Calculator: Dynamic delay penalty breakdown calculating passenger compensation, hotel accommodation, and food voucher liabilities.
  <img width="932" height="140" alt="image" src="https://github.com/user-attachments/assets/b041b783-6e81-493c-a486-fbfbd2e05c37" />

* Visual Breakdown: It is a horizontal bar chart breaking down the aircraft's weight distribution into three core categories: Passengers, Baggage and Available Cargo.
<img width="646" height="374" alt="image" src="https://github.com/user-attachments/assets/e22b00c1-a90d-4b8c-bbbd-6fcf13d8113b" />


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
   git clone <https://github.com/chloe-wanjiru/skycargo-intelligence.git>
   cd <skycargo-intelligence>

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
