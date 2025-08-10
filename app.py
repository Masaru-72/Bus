import random
import pandas as pd
import numpy as np
import math
import heapq
from datetime import timedelta, datetime
import csv
from flask import Flask, render_template, request, jsonify
from sklearn.cluster import KMeans
from functools import partial
import copy
import googlemaps 
from functools import lru_cache
import os
import json

# --- Configuration ---
WALKING_SPEED_KMPH = 5
TRANSFER_PENALTY_MINUTES = 7 
WALK_TRANSFER_MAX_DISTANCE_KM = 2.0 
CACHE_DIR = "network_cache"

app = Flask(__name__)
gmaps = googlemaps.Client(key="YOUR_API_KEY")

# --- In-memory cache for the generated network ---
network_cache = {
    "all_stops": [],
    "solved_routes": [], 
    "distance_matrix_m": None, 
    "duration_matrix_s": None,
    "id_to_stop": {},
    "strategies": {} 
}

# --- Data Loading and Helper Functions (Unchanged) ---
def load_stop_data(filepath="location.csv"):
    stops = []
    COMMERCIAL_ZONE_LATITUDE_BOUNDARY = 37.45
    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            lat = float(row['lat'])
            area_type = 'commercial' if lat > COMMERCIAL_ZONE_LATITUDE_BOUNDARY else 'residential'
            stops.append({"id": i + 1, "name": row['name'], "lat": lat, "lng": float(row['lng']), "area_type": area_type})
    return stops

STOPS_BASE_DATA = load_stop_data()
STOP_NAME_TO_ID = {stop['name']: stop['id'] for stop in STOPS_BASE_DATA}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def build_google_matrices(stops):
    max_id = max(abs(s['id']) for s in stops) if stops else 0
    dist_matrix = np.zeros((max_id + 1, max_id + 1))
    dura_matrix = np.zeros((max_id + 1, max_id + 1))
    
    id_to_stop_map = {s['id']: s for s in stops}
    stop_coords = [(s['lat'], s['lng']) for s in stops]

    batch_size = 10
    for i in range(0, len(stops), batch_size):
        origin_batch = stop_coords[i:i + batch_size]
        origin_stops_batch = stops[i:i+batch_size]
        for j in range(0, len(stops), batch_size):
            dest_batch = stop_coords[j:j + batch_size]
            dest_stops_batch = stops[j:j+batch_size]
            try:
                response = gmaps.distance_matrix(origin_batch, dest_batch, mode="driving")
                for row_idx, row_data in enumerate(response['rows']):
                    origin_id = origin_stops_batch[row_idx]['id']
                    for col_idx, element in enumerate(row_data['elements']):
                        dest_id = dest_stops_batch[col_idx]['id']
                        if element['status'] == 'OK':
                            dist_matrix[abs(origin_id)][abs(dest_id)] = element['distance']['value']
                            dura_matrix[abs(origin_id)][abs(dest_id)] = element['duration']['value']
                        else:
                            fallback_dist_km = haversine(id_to_stop_map[origin_id]['lat'], id_to_stop_map[origin_id]['lng'], id_to_stop_map[dest_id]['lat'], id_to_stop_map[dest_id]['lng'])
                            dist_matrix[abs(origin_id)][abs(dest_id)] = fallback_dist_km * 1000
                            dura_matrix[abs(origin_id)][abs(dest_id)] = (fallback_dist_km / 30) * 3600
            except Exception as e:
                print(f"Google Maps API Error: {e}")
                for origin_stop in origin_stops_batch:
                    for dest_stop in dest_stops_batch:
                        fallback_dist_km = haversine(origin_stop['lat'], origin_stop['lng'], dest_stop['lat'], dest_stop['lng'])
                        dist_matrix[abs(origin_stop['id'])][abs(dest_stop['id'])] = fallback_dist_km * 1000
                        dura_matrix[abs(origin_stop['id'])][abs(dest_stop['id'])] = (fallback_dist_km / 30) * 3600
    return dist_matrix, dura_matrix

def simulate_dynamic_data(stops, sim_time_str="09:00"):
    sim_time = datetime.strptime(sim_time_str, "%H:%M")
    is_peak = "07:00" <= sim_time_str <= "09:30"
    simulated = []
    for s in stops:
        new = s.copy()
        # MODIFIED: Increased demand contrast for more distinct network behavior
        if is_peak and s['area_type'] == 'residential':
            new['demand'] = random.randint(15, 30) # Prioritize residential areas heavily during peak
        else:
            new['demand'] = random.randint(1, 5)   # Lower demand for other cases
        
        new['time'] = (sim_time + timedelta(minutes=random.randint(20, 75))).strftime("%H:%M:%S")
        simulated.append(new)
    return simulated, is_peak

def compute_cost(route, duration_matrix_s, distance_matrix_m, stations, **kwargs):
    total_distance_m, total_ride_time_s, penalty = 0, 0, 0
    customer_stop_times = [datetime.strptime(s['time'], "%H:%M:%S") for s in stations if s.get('area_type') != 'depot']
    if not customer_stop_times: return float('inf')
    first_time = min(customer_stop_times)
    scheduled_times = {s['id']: (first_time if s.get('area_type') == 'depot' else datetime.strptime(s['time'], "%H:%M:%S")) for s in stations}
    current_time = scheduled_times[route[0]['id']]
    for i in range(len(route) - 1):
        from_stop, to_stop = route[i], route[i+1]
        dist_m = distance_matrix_m[abs(from_stop['id'])][abs(to_stop['id'])]
        travel_seconds = duration_matrix_s[abs(from_stop['id'])][abs(to_stop['id'])]
        
        total_distance_m += dist_m
        current_time += timedelta(seconds=travel_seconds)
        if to_stop['id'] in scheduled_times:
            deviation_minutes = abs((current_time - scheduled_times[to_stop['id']]).total_seconds()) / 60
            if deviation_minutes > 5: penalty += 100 * (deviation_minutes - 5)
        total_ride_time_s += travel_seconds * max(0, len(route) - i - 2)
        
    return (total_distance_m / 1000) * 1.0 + (total_ride_time_s / 60) * 0.5 + penalty

def find_best_route_for_cluster(cost_fn, stations):
    if len(stations) < 2:
        return None

    num_stops = len(stations)

    # === Generate diversified initial population ===
    def generate_solution():
        middle = list(range(1, num_stops))
        random.shuffle(middle)
        return [0] + middle

    # === Advanced mutation operator ===
    def mutate(route):
        new_route = route[:]
        op = random.choice(["swap", "reverse", "two_opt"])
        
        if op == "swap":
            i, j = random.sample(range(1, len(new_route)), 2)
            new_route[i], new_route[j] = new_route[j], new_route[i]
        
        elif op == "reverse":
            i, j = sorted(random.sample(range(1, len(new_route)), 2))
            new_route[i:j+1] = reversed(new_route[i:j+1])
        
        elif op == "two_opt":
            i, j = sorted(random.sample(range(1, len(new_route)), 2))
            if i < j:
                new_route = new_route[:i] + new_route[i:j+1][::-1] + new_route[j+1:]
        
        return new_route

    # === Local tweak (small hill-climb step) ===
    def local_tweak(route):
        return mutate(route)  # could be upgraded later

    # === Parameters ===
    num_bats = 30
    max_iter = 100
    a = 0.9
    gamma = 0.9

    # === Memoization ===
    @lru_cache(maxsize=None)
    def eval_fitness(tuple_route):
        return cost_fn(route=[stations[i] for i in tuple_route + (0,)])

    bats = [generate_solution() for _ in range(num_bats)]
    fitness = [eval_fitness(tuple(b + [0])) for b in bats]
    best_idx = min(range(num_bats), key=lambda i: fitness[i])
    best_route_indices = bats[best_idx]
    best_cost = fitness[best_idx]

    loudness = [1.0] * num_bats
    pulse_rate = [0.5] * num_bats

    for t in range(max_iter):
        for i in range(num_bats):
            base = best_route_indices if random.random() > pulse_rate[i] else bats[i]
            candidate = mutate(base)
            candidate_cost = eval_fitness(tuple(candidate + [0]))

            # Accept the new candidate?
            if candidate_cost < fitness[i] or random.random() < loudness[i]:
                candidate = local_tweak(candidate)
                bats[i] = candidate
                fitness[i] = eval_fitness(tuple(candidate + [0]))
                loudness[i] *= a
                pulse_rate[i] *= (1 - math.exp(-gamma * t))

                if fitness[i] < best_cost:
                    best_cost = fitness[i]
                    best_route_indices = candidate

    return best_cost, [stations[i] for i in best_route_indices + [0]]

# --- Network Generation Strategy Implementations ---

def _generate_routes_from_clusters(cluster_labels, num_buses, customer_stops, central_hub, dist_matrix_m, dura_matrix_s, simulation_start_time, demand_map, id_to_stop_map):
    solved_routes = []
    for i in range(num_buses):
        cluster_stops_data = [customer_stops[j] for j, label in enumerate(cluster_labels) if label == i]

        if not cluster_stops_data: continue

        subproblem_stops = [central_hub] + cluster_stops_data
        cost_solver = partial(compute_cost, duration_matrix_s=dura_matrix_s, distance_matrix_m=dist_matrix_m, stations=subproblem_stops)
        
        best_route_tuple = find_best_route_for_cluster(cost_solver, subproblem_stops)
        
        if best_route_tuple:
            cost, final_route_stops = best_route_tuple
            final_route_ids = [s['id'] for s in final_route_stops]
            schedule, current_time = [], simulation_start_time
            
            for j, stop_id in enumerate(final_route_ids):
                if j > 0:
                    prev_stop_id = final_route_ids[j-1]
                    travel_seconds = dura_matrix_s[abs(prev_stop_id)][abs(stop_id)]
                    current_time += timedelta(seconds=travel_seconds)
                
                schedule.append({
                    "stop_id": stop_id, "stop_name": id_to_stop_map[stop_id]['name'],
                    "eta": current_time.strftime("%H:%M:%S")
                })

            solved_routes.append({
                "id": i, "stop_ids": final_route_ids, "cost": round(cost, 2), "schedule": schedule
            })
    return solved_routes

def _generate_balanced_network(customer_stops, num_buses, central_hub, dist_matrix_m, dura_matrix_s, simulation_start_time, demand_map, id_to_stop_map):
    print("--- Generating: BALANCED Network ---")
    customer_coords = np.array([[s['lat'], s['lng']] for s in customer_stops])
    kmeans = KMeans(n_clusters=num_buses, random_state=42, n_init='auto').fit(customer_coords)
    return _generate_routes_from_clusters(kmeans.labels_, num_buses, customer_stops, central_hub, dist_matrix_m, dura_matrix_s, simulation_start_time, demand_map, id_to_stop_map)

def _generate_demand_network(customer_stops, num_buses, central_hub, dist_matrix_m, dura_matrix_s, simulation_start_time, demand_map, id_to_stop_map):
    print("--- Generating: DEMAND-FOCUSED Network ---")
    customer_coords = np.array([[s['lat'], s['lng']] for s in customer_stops])
    sample_weights = np.array([s['demand'] for s in customer_stops]) # Use demand as weight
    kmeans = KMeans(n_clusters=num_buses, random_state=42, n_init='auto').fit(customer_coords, sample_weight=sample_weights)
    return _generate_routes_from_clusters(kmeans.labels_, num_buses, customer_stops, central_hub, dist_matrix_m, dura_matrix_s, simulation_start_time, demand_map, id_to_stop_map)

def _generate_zonal_network(customer_stops, num_buses, central_hub, dist_matrix_m, dura_matrix_s, simulation_start_time, demand_map, id_to_stop_map):
    print("--- Generating: ZONAL (Hub & Spoke) Network ---")
    customer_coords = np.array([[s['lat'], s['lng']] for s in customer_stops])
    
    if len(customer_coords) < num_buses:
        return []
        
    # Use KMeans to create geographical zones (clusters) based on location.
    kmeans = KMeans(n_clusters=num_buses, random_state=42, n_init='auto').fit(customer_coords)
    
    # For each zone, find the optimal ONE-WAY route using the Bat Algorithm.
    # This reuses the same helper function as the 'balanced' network to create one-way loops.
    return _generate_routes_from_clusters(
        kmeans.labels_, 
        num_buses, 
        customer_stops, 
        central_hub, 
        dist_matrix_m, 
        dura_matrix_s, 
        simulation_start_time, 
        demand_map, 
        id_to_stop_map
    )

# --- Main Flask Routes ---

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/generate-network', methods=['POST'])
def generate_network_api():
    data = request.get_json()
    sim_time_str = data.get('time', '09:00')
    num_buses = int(data.get('num_buses', 4))
    
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
    
    simulation_start_time = datetime.strptime(sim_time_str, "%H:%M")
    customer_stops, is_peak = simulate_dynamic_data(STOPS_BASE_DATA, sim_time_str)
    demand_map = {stop['id']: stop['demand'] for stop in customer_stops}

    if not customer_stops or num_buses <= 0: return jsonify({"all_stops": [], "networks": {}})
    if num_buses > len(customer_stops): num_buses = len(customer_stops)
    
    customer_coords = np.array([[s['lat'], s['lng']] for s in customer_stops])
    avg_lat = np.mean(customer_coords[:, 0])
    avg_lng = np.mean(customer_coords[:, 1])
    central_hub = {"id": 0, "name": "Central Station Hub", "lat": avg_lat, "lng": avg_lng, "area_type": "depot", "demand": 0}

    all_network_stops = [central_hub] + customer_stops
    dist_matrix_m, dura_matrix_s = build_google_matrices(all_network_stops)
    id_to_stop_map = {s['id']: s for s in all_network_stops}

    network_cache.update({
        "all_stops": all_network_stops, "id_to_stop": id_to_stop_map,
        "distance_matrix_m": dist_matrix_m, "duration_matrix_s": dura_matrix_s,
        "strategies": {}, "solved_routes": []
    })

    all_networks = {
        'balanced': _generate_balanced_network(customer_stops, num_buses, central_hub, dist_matrix_m, dura_matrix_s, simulation_start_time, demand_map, id_to_stop_map),
        'demand': _generate_demand_network(customer_stops, num_buses, central_hub, dist_matrix_m, dura_matrix_s, simulation_start_time, demand_map, id_to_stop_map),
        'zonal': _generate_zonal_network(customer_stops, num_buses, central_hub, dist_matrix_m, dura_matrix_s, simulation_start_time, demand_map, id_to_stop_map)
    }

    all_strategies_data = {}
    all_routes_for_pathfinding_cache = []
    route_id_counter = 0

    for strategy_name, routes in all_networks.items():
        strategy_graph = {}
        for route in routes:
            route['id'] = route_id_counter
            all_routes_for_pathfinding_cache.append(route)
            
            for i in range(len(route['stop_ids']) - 1):
                u, v = route['stop_ids'][i], route['stop_ids'][i+1]
                if u not in strategy_graph: strategy_graph[u] = []
                strategy_graph[u].append({'to': v, 'route_id': route['id']})
            route_id_counter += 1

        all_strategies_data[strategy_name] = {
            'routes': routes,
            'graph': strategy_graph
        }

    network_cache['strategies'] = all_strategies_data
    network_cache['solved_routes'] = all_routes_for_pathfinding_cache
    
    return jsonify({"all_stops": all_network_stops, "networks": all_networks})

def _find_fastest_path(start_id, end_id, temp_bus_graph, transfer_penalty_multiplier=1.0, walk_penalty_multiplier=1.0):
    all_stops_in_network = network_cache['id_to_stop']
    dist = {stop_id: float('inf') for stop_id in all_stops_in_network}
    dist[start_id] = 0
    pq = [(0, start_id, [])]
    
    while pq:
        time, u, path_log = heapq.heappop(pq)
        
        if time > dist.get(u, float('inf')): continue
        if u == end_id: return time, path_log

        for leg in temp_bus_graph.get(u, []):
            v = leg['to']
            if v in dist:
                travel_time_minutes = network_cache['duration_matrix_s'][abs(u)][abs(v)] / 60
                new_time = time + travel_time_minutes
                if path_log and path_log[-1]['type'] == 'bus' and path_log[-1]['route_id'] != leg['route_id']:
                    new_time += (TRANSFER_PENALTY_MINUTES * transfer_penalty_multiplier)
                if new_time < dist.get(v, float('inf')):
                    dist[v] = new_time
                    heapq.heappush(pq, (new_time, v, path_log + [{'type': 'bus', 'from': u, 'to': v, 'route_id': leg['route_id']}]))
        
        u_stop = all_stops_in_network[u]
        for v_id in all_stops_in_network:
            if u == v_id or v_id <= 0: continue
            v_stop = all_stops_in_network[v_id]
            walk_dist_km = haversine(u_stop['lat'], u_stop['lng'], v_stop['lat'], v_stop['lng'])
            
            if walk_dist_km <= WALK_TRANSFER_MAX_DISTANCE_KM:
                walk_time_minutes = (walk_dist_km / WALKING_SPEED_KMPH) * 60
                new_time = time + (walk_time_minutes * walk_penalty_multiplier)
                if new_time < dist.get(v_id, float('inf')):
                    dist[v_id] = new_time
                    heapq.heappush(pq, (new_time, v_id, path_log + [{'type': 'walk', 'from': u, 'to': v_id}]))
    return float('inf'), []

@app.route('/api/find-journey', methods=['POST'])
def find_journey_api():
    data = request.get_json()
    start_name = data.get('start')
    end_name = data.get('end')
    stopover_names = data.get('stopovers', [])
    start_time_str = data.get('start_time', '09:00')

    try:
        start_id = STOP_NAME_TO_ID[start_name]
        end_id = STOP_NAME_TO_ID[end_name]
        stopover_ids = [STOP_NAME_TO_ID[name] for name in stopover_names]
    except KeyError:
        return jsonify({"error": "Invalid stop name provided."}), 400

    if not network_cache.get('solved_routes'):
        return jsonify({"error": "Network not generated yet."}), 500

    waypoints = [start_id] + stopover_ids + [end_id]
    
    journeys_found = []
    all_strategies = network_cache.get('strategies', {})

    for strategy_name, strategy_data in all_strategies.items():
        full_path_log = []
        is_viable_path = True
        
        for i in range(len(waypoints) - 1):
            leg_start_id, leg_end_id = waypoints[i], waypoints[i+1]
            _, leg_path = _find_fastest_path(
                leg_start_id, 
                leg_end_id, 
                strategy_data['graph'], 
                transfer_penalty_multiplier=1.0, 
                walk_penalty_multiplier=1.0
            )
            
            if not leg_path:
                is_viable_path = False
                break
            full_path_log.extend(leg_path)

        if not is_viable_path:
            continue
            
        current_time_dt = datetime.strptime(start_time_str, "%H:%M")
        segments = []
        current_bus_segment = None
        
        initial_journey_time = copy.deepcopy(current_time_dt)
        processing_time_dt = copy.deepcopy(current_time_dt)

        for i, step in enumerate(full_path_log):
            from_stop = copy.deepcopy(network_cache['id_to_stop'].get(step['from']))
            to_stop = copy.deepcopy(network_cache['id_to_stop'].get(step['to']))
            if not from_stop or not to_stop: continue

            if step['type'] == 'bus' and i > 0:
                prev_step = full_path_log[i-1]
                if prev_step['type'] == 'bus' and prev_step['route_id'] != step['route_id']:
                    processing_time_dt += timedelta(minutes=TRANSFER_PENALTY_MINUTES)
            
            from_stop['eta'] = processing_time_dt.strftime("%H:%M")

            if step['type'] == 'bus':
                travel_seconds = network_cache['duration_matrix_s'][abs(step['from'])][abs(step['to'])]
                processing_time_dt += timedelta(seconds=travel_seconds)
            else:
                walk_dist_km = haversine(from_stop['lat'], from_stop['lng'], to_stop['lat'], to_stop['lng'])
                walk_time_minutes = (walk_dist_km / WALKING_SPEED_KMPH) * 60
                processing_time_dt += timedelta(minutes=walk_time_minutes)
            
            to_stop['eta'] = processing_time_dt.strftime("%H:%M")

            if step['type'] == 'bus':
                if not current_bus_segment or current_bus_segment['route_id'] != step['route_id']:
                    if current_bus_segment: segments.append(current_bus_segment)
                    current_bus_segment = {'type': 'bus', 'route_id': step['route_id'], 'stops': [from_stop]}
                current_bus_segment['stops'].append(to_stop)
            else:
                if current_bus_segment:
                    segments.append(current_bus_segment)
                    current_bus_segment = None
                segments.append({'type': 'walk', 'stops': [from_stop, to_stop]})
        
        if current_bus_segment:
            segments.append(current_bus_segment)

        total_journey_time_minutes = (processing_time_dt - initial_journey_time).total_seconds() / 60

        for seg in segments:
            if seg['type'] == 'bus':
                unique_stops = {stop['id']: stop for stop in seg['stops']}
                seg['stops'] = sorted(list(unique_stops.values()), key=lambda x: x['eta'])

        for seg in segments:
            from_s, to_s = seg['stops'][0], seg['stops'][-1]
            
            if seg['type'] == 'walk':
                seg['description'] = f"Walk from {from_s['name']} to {to_s['name']}"
            else: # It's a bus
                seg['description'] = f"Take Bus {seg['route_id'] + 1} from {from_s['name']} to {to_s['name']}"

        journeys_found.append({
            'id': 0, 
            'name': f"{strategy_name.capitalize()} Network", 
            'time': round(total_journey_time_minutes, 1),
            'segments': segments
        })

    journeys_found.sort(key=lambda j: j['time'])
    for i, journey in enumerate(journeys_found):
        journey['id'] = i + 1

    return jsonify({"journeys": journeys_found})


if __name__ == '__main__':
    app.run(debug=True, port=5001)