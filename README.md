# Eleven Hackathon Project

This repository contains the source code for the Eleven Hackathon challenge. The application is a recommender system demonstrating both cold-start and standard recommendation scenarios.

## Repository Contents

- **Combined.ipynb**: Notebook containing the main logic for client recommendations.
- **cold_start.ipynb**: Notebook demonstrating the cold-start simulator logic.
- **app.py**: The main entry point for the Streamlit web application.
- **src/**: Source code directory containing helper modules for data loading, features, and UI components.
- **eleven_theme.css**: Custom CSS for styling the Streamlit application.

## Prerequisites

Before running the application, you must set up the data:

1.  Create a folder named `data` in the root directory of this repository.
2.  Place all required datasets into this `data` folder.

## Instructions

To run the application, follow these steps in order:

1.  **Run the Notebooks**: You must run the notebooks to generate the necessary underlying data structures and models.
    *   First, run `Combined.ipynb`.
    *   Then, run `cold_start.ipynb`.

2.  **Run the Application**: Once the notebooks have completed and the data is ready, launch the Streamlit app by running the following command in your terminal:

    ```bash
    streamlit run app.py
    ```
