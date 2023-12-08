import streamlit as st

st.set_page_config(
    page_title="Prediction Tool",
    page_icon="🫡",
    menu_items={
        "About": """This web application is a project that aims to assist AllLife Bank in identifying potential customers 
        likely to accept personal loan offers. It is based on a dataset of customer demographics and financial behavior."""
    },
)


st.title("About Me 🫡")


st.image("images/peter.jpg", width=200)
st.markdown(
    """
    Hello! I'm Peter Imoukhuede, an aspiring Data Scientist and Master's student at Michigan State University, with a keen interest in technology 🖥️ and Formula 1 🏎️. My background includes an electrical and electronics engineering degree 🎓 from Covenant University, Nigeria, and a postgraduate degree in Data Science and Business Analysis from the University of Texas at Austin.

    I am skilled in R, Python, SQL, Tableau, TensorFlow, and NLP, I've worked as an Operations Analyst 📈 at Thor Explorations, leveraging data science in practical business scenarios.

    I've led projects in credit card churn prediction, seedling classification, airline sentiment analysis, and travel package prediction, using advanced techniques like Artificial Neural Networks.

    Beyond data science, I'm intrigued by autonomous transportation 🚗 and affective computing. Outside work, I enjoy music 🎵, Formula 1 🏁, and good movies 🎬
    """
)
