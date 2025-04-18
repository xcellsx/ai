# Emotional Journal - React Application

## Description

Emotional Journal is a web application designed to help users explore and understand their emotions through daily journaling. Users can write entries, receive an AI-powered analysis of the inferred emotion, visualize their emotional trends over time on a calendar, and view aggregated reports. The goal is to promote self-reflection and emotional awareness.

This application uses client-side Local Storage for data persistence, meaning all journal entries are stored directly in your browser.

## Features

* **Journal Entry:** Create new journal entries using a simple text editor.
* **Emotion Analysis:** Submit entries to a backend AI model (via Flask API) to get an inferred emotion classification (e.g., sadness, joy, anger).
* **History Calendar:** View a monthly calendar where days are color-coded based on the most frequent emotion logged for that day.
* **Month Navigation:** Navigate between previous and next months in the history view.
* **Daily Entry Modal:** Click on a date in the calendar to view all entries logged for that specific day, along with emotion counts.
* **Reports:** View summary visualizations:
    * A heatmap showing the relationship between total emotion counts and inferred depression levels.
    * A bar chart displaying the total count for each detected emotion.
* **Data Persistence:** Entries and analysis results are saved locally in the browser's Local Storage.
* **UI:** Minimalist design with a fixed header, consistent navigation, and an animated pastel gradient background.

## Technology Stack

* **Frontend:**
    * React (~v19.1.0 - see installation notes)
    * React Router (`react-router-dom`) for page navigation.
    * Styled Components (`styled-components`) for component styling and theming (including buttons, animated background).
    * ApexCharts (`react-apexcharts`, `apexcharts`) for heatmap and bar chart visualizations.
* **Backend:**
    * Python Flask for the API server.
    * Flask-CORS for handling cross-origin requests from the React frontend.
    * Hugging Face `transformers` library for loading and using the sentiment analysis model.
    * PyTorch (`torch`) as the backend for the transformers model.
    * NumPy (often a dependency for ML tasks).
* **Storage:**
    * Browser Local Storage (Client-Side).

## Setup and Installation

### Prerequisites

* Node.js and npm (or yarn) installed.
* Python (3.7+ recommended) and pip installed.

### Backend Setup (Flask API)

1.  **Navigate:** Open your terminal and navigate to the directory containing `app.py` (e.g., `cd path/to/ai\ project/backend`).
2.  **Create Virtual Environment:** It's highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    ```
3.  **Activate Virtual Environment:**
    * macOS/Linux: `source venv/bin/activate`
    * Windows (cmd): `venv\Scripts\activate`
    * Windows (PowerShell): `venv\Scripts\Activate.ps1`
4.  **Install Python Dependencies:**
    ```bash
    pip install Flask Flask-Cors transformers torch numpy
    ```
    * **Note on PyTorch (`torch`):** For optimal performance (especially with a GPU), you might need a specific version of PyTorch corresponding to your CUDA version. Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for the recommended installation command for your system. The command above installs the standard CPU/GPU version.
5.  **Download/Place Model:** Ensure you have the pre-trained Hugging Face model saved. The code expects it at `./results/bert_uncased_L-2_H-128_A-2-finetuned-emotion/best_model` relative to `app.py`. **Verify the `MODEL_PATH` variable inside `app.py` points to the correct location.**
6.  **(Optional) Create `requirements.txt`:** You can run `pip freeze > requirements.txt` to save the dependencies for easier installation later (`pip install -r requirements.txt`).

### Frontend Setup (React App)

1.  **Navigate:** Open another terminal window/tab and navigate to your React project directory:
    ```bash
    cd path/to/ai\ project/my-react-app
    ```
2.  **Install Node Dependencies:** This command installs React, React Router, Styled Components, ApexCharts, and all other necessary packages listed in your `package.json`.
    ```bash
    npm install
    ```
    * *Alternatively, if you use Yarn:*
        ```bash
        yarn install
        ```
3.  **Dependency Overview:** Key libraries installed by the command above (or added previously) include:
    * `react`, `react-dom`
    * `react-router-dom`
    * `styled-components`
    * `react-apexcharts`
    * `apexcharts`

4.  **React 19 Warning:** You are using React `~19.1.0`. Some libraries, like `react-apexcharts`, might not have explicitly listed compatibility with React 19 yet and could show **peer dependency warnings** during installation. If you encounter runtime errors:
    * Check the library's documentation/GitHub for React 19 support updates.
    * Consider downgrading React to the latest v18 (`npm install react@^18.2.0 react-dom@^18.2.0`).
    * As a last resort, you *might* bypass the warning with `npm install --legacy-peer-deps`, but this is risky and not recommended for stability.

## Running the Application

You need to run both the Backend and Frontend servers simultaneously.

1.  **Run Backend Server:**
    * Navigate to the backend directory (`cd path/to/ai\ project/backend`).
    * Activate the virtual environment (e.g., `source venv/bin/activate`).
    * Run the Flask app: `python app.py`
    * Keep this terminal running. It will likely serve on `http://localhost:5000`.
2.  **Run Frontend Server:**
    * Navigate to the frontend directory (`cd path/to/ai\ project/my-react-app`).
    * Run the React development server: `npm start` (or `yarn start`).
    * This should automatically open the application in your browser, usually at `http://localhost:3000`.
    * Keep this terminal running.

Now you can access the Emotional Journal app in your browser!

## Usage

1.  The application opens on the **Home** page.
2.  Use the navigation buttons (`Home`, `Journal Entry`, `View History`, `View Report`) in the header or on the pages to move between sections.
3.  Go to **Journal Entry** to write your thoughts and click "Submit Entry". The app will send the text to the backend for analysis and display the result. Entries are saved automatically to your browser's Local Storage.
4.  Go to **View History** to see a calendar view. Days with entries are colored based on the most frequent emotion logged that day. Use the arrows to navigate months. Click on a specific date to open a modal showing all entries and emotion counts for that day.
5.  Go to **View Report** to see visualizations based on *all* your saved entries, including a heatmap correlating emotions with inferred depression levels and a bar chart showing total emotion counts.
6.  Data is stored **only in your current browser**. Clearing browser data will remove your journal history.

## Disclaimer

**This application is for informational and self-reflection purposes only. It is not a diagnostic tool and cannot replace professional medical or psychological advice, diagnosis, or treatment. If you are experiencing significant emotional distress, please consult with a qualified healthcare professional.**
