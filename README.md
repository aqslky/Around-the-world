# Around the World Project
This project solves a computational problem involving world cities, graph theory, and shortest-path algorithms.

# Goal
Starting in London and always traveling east, can one travel around the world and return to London in 80 days?

# Rules
- Every city is connected to its 3 nearest neighbors
- Travel times:
- - 2 hours to the nearest city
- - 4 hours to the second nearest city
- - 8 hours to the third nearest city
- - Add +2 hours if destination is in a different country
- - Add +2 hours if destination has population > 200,000

# Dataset 
The project uses the World Cities Database available on Kaggle:
https://www.kaggle.com/datasets/max-mind/world-cities-database?select=worldcitiespop.csv

Columns include:
- Country
- City name
- Latitude / Longitude
- Population
- Region