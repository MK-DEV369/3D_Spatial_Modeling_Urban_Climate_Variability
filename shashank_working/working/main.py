import joblib
import pickle
import sys
import copy
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import your custom modules
from arima import get_forecast_up_to_date
from add_road import add_custom_edge, remove_custom_edge
from calc import calc_pollutions

app = Flask(__name__)
CORS(app)

# --- Global Configurations & Model Loading ---
CURRENT_VEHICLES = 12112902

try:
    print("Loading models and graph...")
    model1 = joblib.load("transport_arima.joblib")
    model2 = joblib.load("non_transport_arima.joblib")
    with open("bengaluru_graph.pkl", "rb") as f:
        G_base = pickle.load(f)
    print("Setup complete. Server is ready.")
except FileNotFoundError:
    print("Pickle file or model file not found.")
    sys.exit(0)
except Exception as e:
    print(f"An error occurred during startup: {e}")
    sys.exit(1)

@app.route('/simulate', methods=['POST'])
def simulate():
    start_time = datetime.now()
    
    # Get data from request body
    data = request.get_json()
    if not data or 'date' not in data:
        return jsonify({"error": "Missing 'date' parameter"}), 400

    date = data.get('date')
    added_edges = data.get('added_edges', [])
    deleted_edges = data.get('deleted_edges', [])

    # Retaining all your original logging statements
    print(f"Starting prediction for date: {date}")
    
    try:
        # Forecast logic
        forecast_df = get_forecast_up_to_date(
            target_date=date, 
            model_list=[model1, model2], 
            model_names=["transport", "non_transport"], 
            last_train_date="2025-09-01"
        )
        
        print(" == Forecast results == ")
        print(forecast_df.iloc[-1])
        
        total_vehicles = forecast_df.iloc[-1]['transport_forecast'] + forecast_df.iloc[-1]['non_transport_forecast']
        print(f"Total vehicles: {int(total_vehicles)}")
        
        ratio = float(total_vehicles) / CURRENT_VEHICLES
        print(f"Ratio: {ratio}")

        # Deepcopy the graph so modifications don't persist across different API calls
        G = copy.deepcopy(G_base)

        print("Starting addition of custom edges...")
        for edge in added_edges:
            # Assuming edge is a list/tuple like [u, v]
            add_custom_edge(G, edge[0], edge[1])

        print("Starting deletion of edges...")
        for edge in deleted_edges:
            remove_custom_edge(G, edge[0], edge[1])

        print("Starting result calculation...")
        results = calc_pollutions(G, ratio)
        
        end_time = datetime.now()
        print(f"Started at time: {start_time}")
        print(f"Ended at time: {end_time}")
        print(f"Took: {end_time - start_time} time")

        # Return the results as JSON
        return jsonify({
            "date": date,
            "total_vehicles": int(total_vehicles),
            "ratio": ratio,
            "results": results,
            "execution_time": str(end_time - start_time)
        })

    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Running on port 5000 by default
    app.run(host='0.0.0.0', port=5000, debug=True)