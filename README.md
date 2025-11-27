# Fantasy Premier League Team Optimiser

A small Python + Streamlit app that fetches live Fantasy Premier League (FPL) data, predicts player points using machine learning, and builds an optimal squad under a given budget.

The app:

- Pulls player stats directly from the official FPL API
- Trains a simple regression model to predict future points
- Selects an optimal team within your budget
- Displays the selected squad and a cost distribution chart

> This is a personal project and is not affiliated with Fantasy Premier League or the Premier League.

---

## Features

- 📡 **Live data**  
  Uses `https://fantasy.premierleague.com/api/bootstrap-static/` to fetch up-to-date player data.

- 📈 **Points prediction**  
  Trains a `LinearRegression` model (from scikit-learn) using:
  - minutes played  
  - goals scored  
  - assists  
  - clean sheets  

- 🧮 **Team optimisation**  
  Builds a value-for-money squad that:
  - Stays under a specified budget  
  - Targets a specific squad size (default: 15 players)  

- 📊 **Visual insights**  
  - Streamlit table of the selected team  
  - Histogram showing the cost distribution of the chosen players  

- 📝 **Optional automation**  
  A scheduled job can re-run the pipeline at an interval and log the selected team to a file.

---

## Tech Stack

- **Python** 3.10+
- **Libraries**
  - `requests` – fetch FPL API data
  - `pandas` – data wrangling
  - `scikit-learn` – points prediction (linear regression)
  - `matplotlib` – basic visualisation
  - `streamlit` – web UI
  - `schedule` – lightweight job scheduling
  - `numpy`, `scipy` – optimisation utilities (if you use linear programming)

See [`requirements.txt`](requirements.txt) for the full list.

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Voy6/Fantasy-League.git
   cd Fantasy-League
