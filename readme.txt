# Dynamic Bus Network Simulator

This Flask application simulates and analyzes different strategies for creating dynamic bus routes based on real-time demand. It allows users to generate bus networks using various models (Balanced, Demand-Focused, Zonal) and then find the optimal journey between any two points, including transfers and walking.

## Features

- **Dynamic Network Generation**: Creates bus routes based on simulated passenger demand for a specific time of day.
- **Multiple Strategies**: Implements and compares three distinct network generation strategies:
    1.  **Balanced**: Routes are created to serve geographically balanced clusters of stops.
    2.  **Demand-Focused**: Clusters are weighted by passenger demand, prioritizing high-traffic areas.
    3.  **Zonal (Hub & Spoke)**: Routes operate as spokes connecting clusters of stops to a central hub.
- **Realistic Travel Times**: Uses the **Google Maps Distance Matrix API** to get real-world driving times and distances between stops.
- **Optimal Journey Planning**: Finds the fastest multi-modal journey (bus and walking) from a start to an end point, including waypoints.
- **Intelligent Pathfinding**: Uses Dijkstra's algorithm to calculate the best path, accounting for travel time, walking transfers, and penalties for switching buses.
- **Web-Based UI**: Provides an interactive interface to generate networks, visualize routes on a map, and plan journeys.

## How It Works

1.  **Data Loading**: The application starts by loading a list of bus stop locations from `location.csv`.
2.  **Network Generation**:
    - When a user requests to generate a network for a given time, the app simulates passenger demand at each stop.
    - It uses the Google Maps API to build a complete distance and duration matrix between all stops.
    - **K-Means clustering** is used to group stops into zones for the number of available buses. The `Demand-Focused` strategy uses passenger demand to weight the clustering algorithm.
    - For each cluster, a **metaheuristic solver (Bat Algorithm)** finds an optimized route (a solution to the Traveling Salesperson Problem) that starts and ends at a central depot.
    - The generated routes for all strategies are cached in memory.
3.  **Journey Finding**:
    - A user selects a start, end, and optional stopover points.
    - The application builds a graph for each network strategy where nodes are stops and edges are bus segments or possible walking paths.
    - **Dijkstra's algorithm** is run on each graph to find the fastest path. The algorithm's cost function includes bus travel time, walking time, and a time penalty for transfers between different bus routes.
    - The best journeys found across all strategies are presented to the user, sorted by total travel time.

## Setup and Installation

**1. Clone the Repository**
```bash
git clone <repository-url>
cd <repository-directory>