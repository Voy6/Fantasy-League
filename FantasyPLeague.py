import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.optimize import linprog
import schedule
import time
import logging
import streamlit
import matplotlib.pyplot as plt


def fetch_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()

    players_df = pd.json_normalize(data["elements"])
    print(f"Original DataFrame shape: {players_df.shape}")

    players_df = players_df[["first_name", "second_name", "team", "now_cost",
                             "total_points", "minutes", "goals_scored", "assists", "clean_sheets"]]

    players_df["now_cost"] = players_df["now_cost"] / 10

    print(f"DataFrame shape after selection: {players_df.shape}")

    players_df = players_df[players_df["minutes"] > 150]

    print(f"DataFrame shape after filtering: {players_df.shape}")

    return players_df


def predict_points(players_df):
    if players_df.empty:
        raise ValueError("No data available for prediction.")

    features = players_df[["minutes", "goals_scored", "assists", "clean_sheets"]]
    target = players_df["total_points"]

    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")

    players_df["predicted_points"] = model.predict(features)

    return players_df


def optimise_team(players_df, budget=100, squad_size=15):
    C = -players_df["predicted_points"].values

    A = [players_df["now_cost"].values]
    B = [budget]

    bounds = [(0, 1) for _ in range(len(players_df))]

    result = linprog(C, A_ub=A, b_ub=B, bounds=bounds, method="highs")

    selected_indices = [i for i, x in enumerate(result.x) if x == 1]
    best_team = players_df.iloc[selected_indices]

    return best_team


logging.basicConfig(filename="fpl_automation.log", level=logging.INFO,
                    format='%(asctime)s - %(message)s')

streamlit.title("Fantasy Premier League Team Optimiser")

streamlit.write("Fetching Data...")
players_df = fetch_data()
if players_df.empty:
    streamlit.error("No players available for the analysis.")
else:
    streamlit.write("Predicting Points...")
    players_df = predict_points(players_df)

    if not players_df.empty:
        streamlit.write("Optimising Team...")
        best_team = optimise_team(players_df)

        if not best_team.empty:
            streamlit.write("### Selected Team")
            streamlit.dataframe(best_team[["first_name", "second_name", "now_cost", "predicted_points"]])

            streamlit.write("### Cost Distribution")
            fig, ax = plt.subplots()
            best_team["now_cost"].hist(ax=ax)
            ax.set_title("Cost Distribution of Selected Team")
            ax.set_xlabel("Cost")
            ax.set_ylabel("Number of Players")
            streamlit.pyplot(fig)
        else:
            streamlit.error("No optimal team be formed.")
    else:
        streamlit.error("No players available after prediction.")


def update_team():
    logging.info("Fetching Data...")
    players_df = fetch_data()
    if players_df.empty:
        print("No players available for the analysis.")
        return

    logging.info("Predicting Points...")
    players_df = predict_points(players_df)
    if players_df.empty:
        print("No players available after prediction.")
        return

    logging.info("Optimising team...")
    best_team = optimise_team(players_df)
    if best_team.empty:
        print("No optimal team can be formed.")
        return

    logging.info("Selected Team:")
    logging.info("\n" + best_team[["first_name", "second_name", "now_cost", "predicted_points"]].to_string(index=False))


schedule.every(10).seconds.do(update_team)

while True:
    schedule.run_pending()
    time.sleep(1)
