#Install these packages if needed.
Package	        Installation Command	   Purpose
streamlit	pip install streamlit	   For creating interactive web dashboard applications
pandas	        pip install pandas	   For working with tabular data
plotly	        pip install plotly	   For interactive charts in Streamlit
matplotlib	pip install matplotlib	   For classic plotting (used in ARIMA visualization)
scikit-learn	pip install scikit-learn   For evaluation metrics: MAE, RMSE, MAPE
XlsxWriter      pip install XlsxWriter     To create and write Excel files


#Instructions for running the dashboard:
#The database file Database test.xlsx must be in the same folder as ABCCompanyFinal.py
#Run the following command in the terminal: python -m streamlit run ABCCompanyFinal.py