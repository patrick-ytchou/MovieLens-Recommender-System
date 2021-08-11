# import app_mf
# import app_cf
# import streamlit as st

# PAGES = {
#     "Matrix Factorization": app_mf,
#     "Collaborative Filtering": app_cf
# }

# st.sidebar.title('Navigation')
# selection = st.sidebar.radio("Go to", list(PAGES.keys()))
# page = PAGES[selection]
# page.app()

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
# app.add_page("Collaborative Filtering", app_cf.app)
app.add_page("Matrix Factorization", app_mf.app)
# app.add_page("Change Metadata", metadata.app)
# app.add_page("Machine Learning", machine_learning.app)
# app.add_page("Data Analysis",data_visualize.app)
# app.add_page("Y-Parameter Optimization",redundant.app)

# The main app
app.run()