from collections import Counter
import pandas as pd

data = [
    {"Day": "D1", "Outlook": "Sunny", "Humidity": "High", "Wind": "Weak", "Play": "No"},
    {"Day": "D2", "Outlook": "Sunny", "Humidity": "High", "Wind": "Strong", "Play": "No"},
    {"Day": "D3", "Outlook": "Overcast", "Humidity": "High", "Wind": "Weak", "Play": "Yes"},
    {"Day": "D4", "Outlook": "Rain", "Humidity": "High", "Wind": "Weak", "Play": "Yes"},
    {"Day": "D5", "Outlook": "Rain", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Day": "D6", "Outlook": "Rain", "Humidity": "Normal", "Wind": "Strong", "Play": "No"},
    {"Day": "D7", "Outlook": "Overcast", "Humidity": "Normal", "Wind": "Strong", "Play": "Yes"},
    {"Day": "D8", "Outlook": "Sunny", "Humidity": "High", "Wind": "Weak", "Play": "No"},
    {"Day": "D9", "Outlook": "Sunny", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Day": "D10", "Outlook": "Rain", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Day": "D11", "Outlook": "Sunny", "Humidity": "Normal", "Wind": "Strong", "Play": "Yes"},
    {"Day": "D12", "Outlook": "Overcast", "Humidity": "High", "Wind": "Strong", "Play": "Yes"},
    {"Day": "D13", "Outlook": "Overcast", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Day": "D14", "Outlook": "Rain", "Humidity": "High", "Wind": "Strong", "Play": "No"},
]

print(pd.DataFrame(data))


play_yes_count = Counter(row["Play"] for row in data if row["Play"] == "Yes").total()
play_count = len(data)
play_yes_prob = play_yes_count / play_count
print("\n\nProbability of the game being played: {0}/{1} = {2}"
      .format(play_yes_count, play_count, round(play_yes_prob, 3)))

overcast_yes_count = Counter(
    row["Outlook"] for row in data if row["Play"] == "Yes" and row["Outlook"] == "Overcast").total()
overcast_yes_prob = overcast_yes_count / play_yes_count
print("Probability of overcast during the game: {0}/{1} = {2}"
      .format(overcast_yes_count, play_yes_count, round(overcast_yes_prob, 3)))

humidity_high_yes_count = Counter(
    row["Humidity"] for row in data if row["Play"] == "Yes" and row["Humidity"] == "High").total()
humidity_high_yes_prob = humidity_high_yes_count / play_yes_count
print("Probability of high humidity during the game: {0}/{1} = {2}"
      .format(humidity_high_yes_count, play_yes_count, round(humidity_high_yes_prob, 3)))

wind_strong_yes_count = Counter(
    row["Wind"]for row in data if row["Play"] == "Yes" and row["Wind"] == "Strong").total()
wind_strong_yes_prob = wind_strong_yes_count / play_yes_count
print("Probability of strong wind during the game: {0}/{1} = {2}"
      .format(wind_strong_yes_count, play_yes_count, round(wind_strong_yes_prob, 3)))


overall_probability = play_yes_prob * overcast_yes_prob * humidity_high_yes_prob * wind_strong_yes_prob
print("\nProbability of the game being played with conditions overcast, high humidity, strong wind:"
      " \n{:.3f} * {:.3f} * {:.3f} * {:.3f} = {:.4f}"
      .format(play_yes_prob, overcast_yes_prob, humidity_high_yes_prob, wind_strong_yes_prob, overall_probability))
