import streamlit as st

# Custom imports 
from multipage import MultiPage
import intro # introduction page for the recommender engine
import app_mf # matrix factorization app

# Set page config
st.set_page_config(layout="wide")

# Create an instance of the app 
app = MultiPage()

# Title of the main page
# st.title("Choose Your Recommender Engine")

# Add all your applications (pages) here
app.add_page("Introduction", intro.app)
app.add_page("Matrix Factorization", app_mf.app)

# The main app
app.run()