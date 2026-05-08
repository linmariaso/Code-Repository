# Smart Home Energy Prediction
Smart Home Energy Prediction is a Python pipeline that emulates sensor data, preprocesses it, and trains ML models to forecast home energy use.

Smart Home Energy Prediction is an end-to-end Python project that simulates smart home sensor readings, routes them through a data pipeline, and trains machine learning models to forecast household energy consumption. It is designed for experimentation: you can run the full six-phase pipeline, skip the MQTT infrastructure, or target a single phase.

## What the project does
The project emulates sensors across five rooms (bedroom, living_room, bathroom, kitchen, studio) and eight appliances over a configurable time window (730 days by default, sampled every 5 minutes). The emulated data flows through preprocessing and feature engineering stages before three ML models are trained, evaluated, and compared.
The three models are:
* Linear Regression — baseline model for interpretable predictions
* Random Forest — ensemble method for capturing non-linear patterns
* MLP (Multi-Layer Perceptron) — neural network for complex feature interactions

### The six phase pipeline
1. Sensor emulation: Generates a synthetic dataset with emulated appliance and sensor readings, runs validation checks, and saves raw CSV data to data/raw/
2. MQTT Infrastructure: Publishes emulated sensor events over MQTT using paho-mqtt and a Mosquitto broker, with a subscriber capturing live messages
3. Preprocessing: Cleans and transforms raw data, engineers features, and produces train/test splits saved to data/processed/
4. ML training and evaluation: Trains all three models, evaluates R², RMSE, and MAE, and runs statistical significance tests between models
5. Feature importance and decomposition: Runs permutation importance and Oh’s decomposition analysis, and saves plots and CSVs to results/
6. Validation: Compares emulated results against real-world datasets (if available in data/validation/) within a 20% tolerance

### Key Technologies
* Python 3 — runtime and scripting
* scikit-learn — model training and evaluation
* paho-mqtt — MQTT client for sensor event publishing
* pandas / numpy / scipy — data manipulation and statistical analysis
* matplotlib / seaborn — visualisation and result plots
* Mosquitto — open-source MQTT broker (required for Phase 2)

## Get started with Smart Home Energy Prediction
Learn how to install dependencies, generate emulated sensor data, train ML models, and run the full six-phase energy prediction pipeline.

This guide walks you through cloning the project, installing dependencies, and running the Smart Home Energy Prediction pipeline. The full pipeline takes you from sensor emulation through ML model training to validation. You can also run individual phases or skip the MQTT step entirely.
​
### Prerequisites
* Python 3.x
* pip
* (Optional) Mosquitto — required only for Phase 2 (MQTT infrastructure)

1. **Clone the repository and install dependencies**

Clone the project and install all Python dependencies from Requirements.txt:

```bash
git clone https://github.com/linmariaso/Code-Repository.git
cd Code-Repository/smart-home-energy-prediction
pip install -r Requirements.txt
```
This installs: numpy, pandas, scipy, scikit-learn, paho-mqtt, matplotlib, seaborn, and jupyter.

2. **Run the full pipeline**

Run all six phases in sequence — sensor emulation, MQTT, preprocessing, ML training, analysis, and validation:
```python
python run_all.py
```
Phase 2 requires a running Mosquitto broker on localhost:1883. If Mosquitto is not installed or you want to skip it, use the --skip-mqtt flag instead.

3. **Run without MQTT (skip Phase 2)**

If you do not have Mosquitto installed, skip Phase 2 and run all other phases using the emulated dataset from Phase 1:
```python
python run_all.py --skip-mqtt
```
This runs phases 1, 3, 4, 5, and 6.

4. **Run a single phase**

To run only one phase, use --only followed by the phase number (1–6):
```python
python run_all.py --only 4
```
This example runs only Phase 4: ML training and evaluation.

5. **Start from a specific phase**

To resume the pipeline from a particular phase (useful if earlier phases have already completed), use --start-from:
```python
python run_all.py --start-from 3
```
This runs phases 3, 4, 5, and 6.
​
### Expected outputs
After a successful run, the pipeline creates the following directories:
| Directory	| Contents |
|--|--|
|data/raw/ |	Raw emulated sensor data as CSV (test_dataset.csv) |
|data/processed/ |	Engineered features and train/test splits |
|results/ |	Trained models, evaluation metrics, importance plots, and decomposition CSVs |

## Installation and environment setup
Install Python dependencies and configure Mosquitto for Smart Home Energy Prediction. Covers pip setup, MQTT broker configuration, and directory structure.

Smart Home Energy Prediction requires Python 3 and a small set of data science libraries. Mosquitto (an MQTT broker) is only needed if you intend to run Phase 2. This page covers all installation steps and the directory structure the pipeline will create.
​
### Python dependencies
All required packages are listed in Requirements.txt:
```bash
python
numpy
pandas
scipy
scikit-learn
paho-mqtt
matplotlib
seaborn
jupyter
```
Install them with:
```python
pip install -r Requirements.txt
```

​
### MQTT broker setup (optional)
Phase 2 of the pipeline publishes and subscribes to sensor data over MQTT. It requires a running Mosquitto broker on localhost:1883.

> If you do not need Phase 2, skip this section entirely and run the pipeline with --skip-mqtt. All other phases work without MQTT.

* **Linux (apt)**
```bash
sudo apt update
sudo apt install mosquitto mosquitto-clients
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```

* **macOS (Homebrew)**
```bash
brew install mosquitto
brew services start mosquitto
```

* **Windows**
```bash
# Download the installer from the official Mosquitto site:
# https://mosquitto.org/download/
#
# After installation, start the broker from the command prompt:
net start mosquitto
```

To verify the broker is running, check that port 1883 is open:
```bash
# Linux / macOS
mosquitto_sub -h localhost -t test &
mosquitto_pub -h localhost -t test -m "ping"
```

> The pipeline will exit with an error at Phase 2 if it cannot reach the broker at localhost:1883. Start Mosquitto before running python run_all.py, or use --skip-mqtt to bypass Phase 2.
​

### Skipping MQTT
To run the full pipeline without MQTT, pass the --skip-mqtt flag:
```python
python run_all.py --skip-mqtt
```

This runs phases 1, 3, 4, 5, and 6 using the emulated dataset generated in Phase 1.
​
### Directory structure
The pipeline creates the following directories automatically on first run:
```
smart-home-energy-prediction/
├── data/
│   ├── raw/              # Raw emulated sensor CSV (test_dataset.csv)
│   └── processed/        # Feature matrix and train/test splits
├── results/              # Trained models, metrics, plots, CSVs
└── config/
    └── settings.py       # All configurable parameters
```

The results/ directory is created by config/settings.py on import. data/raw/ and data/processed/ are created automatically when Phase 1 and Phase 3 run respectively.
​
### Configuration
Key parameters in config/settings.py you may want to adjust:
| Setting |	Default |	Description |
| -- | -- | -- |
| days_to_generate |	730 |	Number of days of sensor data to emulate |
| sample_freq |	5 |	Sampling frequency in minutes |
| mqtt_broker |	'localhost' |	MQTT broker hostname |
| mqtt_port |	1883 |	MQTT broker port |
| test_ratio	| 0.2 |	Fraction of data held out for testing |
| r_seed |	42 |	Random seed for reproducibility |

### Emulation parameters
These settings control how synthetic sensor data is generated.

| Setting |	Default |	Description |
| -- | -- | -- |
| rooms |	['bedroom','living_room','bathroom','kitchen','studio'] |	Rooms for which sensor readings are generated |
| days_to_generate |	730 |	Total days of synthetic data to produce (two years) |
| sample_freq |	5 |	Sampling frequency in minutes for the main dataset |
| start_time	| datetime(2023,1,1,0,0) |	Start timestamp for the generated time series |
| intervals |	days_to_generate * 24 * (60 // sample_freq) |	Total number of time steps, derived automatically |
| anomaly_rate |	0.20 |	Overall anomaly injection probability |
| anomaly_rate_active |	0.10 |	Anomaly probability when an appliance is active (power > 10 W) |
| anomaly_rate_inactive |	0.015 |	Phantom-load anomaly probability when an appliance is off |
| r_seed |	42 |	NumPy random seed for reproducible data generation |

> Set r_seed to a fixed integer before generating data. Every run with the same seed produces identical output, which is essential for reproducible experiments and fair model comparisons.

### Sensor Intervals
Each sensor type publishes readings at its own interval over MQTT. Values are in seconds.

| Setting |	Default |	Sensor |
| -- | -- | -- |
| temp_interval |	30 |	Temperature sensor polling interval|
| hum_interval |	30 |	Humidity sensor polling interval|
| occ_interval |	10 |	Occupancy sensor polling interval|
| sp_interval |	5 |	Smart plug power sensor polling interval|

### MQTT Settings

| Setting |	Default |	Description |
| -- | -- | -- |
| mqtt_broker |	'localhost' |	Hostname or IP address of the MQTT broker |
| mqtt_port |	1883 |	MQTT broker port (standard unencrypted port) |
| mqtt_topic_prefix	| 'home' |	Prefix prepended to all MQTT topic paths |

### Appliances and rooms
```python
rooms = ['bedroom', 'living_room', 'bathroom', 'kitchen', 'studio']

appliances = [
    'washing_machine', 'refrigerator', 'television', 'microwave',
    'kettle', 'computer', 'dishwasher', 'lighting'
]
```
### Machine Learning Settings

| Setting	| Default |	Description |
| -- | -- | -- |
|test_ratio	| 0.2	| Fraction of data reserved for the held-out test set |
|folds	| 5	| Number of cross-validation folds used in GridSearchCV |
​
### Evaluation targets

These thresholds define what counts as a passing model. The evaluation phase compares each trained model’s metrics against them.

| Setting	| Default	| Description |
| -- | -- | -- |
| r2_target |	0.70 |	Minimum acceptable R² score |
| rmse_target |	100	| Maximum acceptable RMSE in watts |
| mae_target	| 70 |	Maximum acceptable MAE in watts |
| significance	| 0.05 |	p-value threshold for statistical significance tests |
| permutation_repeats |	50	| Number of repeats for permutation feature importance |
​
### Validation
Setting	Default	Description
validation_tolerance	0.20	Allowed relative deviation (±20%) when cross-checking emulated sensor values against expected ranges
​
Full file
```python
config/settings.py
# Configuration File
import os
import numpy as np
from datetime import datetime

# Output directory
output_dir = 'results/'

# Emulation Parameters
rooms = ['bedroom','living_room','bathroom','kitchen','studio']
days_to_generate = 730
sample_freq = 5  # minutes
start_time = datetime(2023,1,1,0,0)
intervals = days_to_generate * 24 * (60 // sample_freq)
anomaly_rate = 0.20
anomaly_rate_active = 0.10
anomaly_rate_inactive = 0.015
r_seed = 42
np.random.seed(r_seed)
os.makedirs(output_dir, exist_ok=True)

# Sensor intervals (seconds)
temp_interval = 30
hum_interval = 30
occ_interval = 10
sp_interval = 5

# Anomalies
an_rate = 0.02

# MQTT
mqtt_broker = 'localhost'
mqtt_port = 1883
mqtt_topic_prefix = 'home'

# Appliances
appliances = ['washing_machine','refrigerator','television','microwave','kettle','computer','dishwasher','lighting']

# Machine Learning settings
test_ratio = 0.2
folds = 5

# Evaluation metrics
r2_target = 0.70
rmse_target = 100
mae_target = 70
significance = 0.05
permutation_repeats = 50

# Validation
validation_tolerance = 0.20
```

## Sensor emulation configuration

Configure temperature, humidity, occupancy, and smart plug sensors — including rooms, appliances, sampling intervals, and anomaly rates for emulation.

The emulation layer generates synthetic readings for four sensor types across every room and appliance defined in config/settings.py. Each sensor module reads its parameters — rooms, appliances, sampling intervals, and anomaly rates — directly from that file, so you only need to edit one place to change the scope of your dataset.
​
### Sensor types

* Temperature sensor
* Humidity sensor
* Occupancy sensor
* Smart plug sensor

​
### Rooms and appliances
The lists below, defined in config/settings.py, determine which rooms and appliances are included in every generated dataset.
```python
config/settings.py
rooms = ['bedroom', 'living_room', 'bathroom', 'kitchen', 'studio']

appliances = [
    'washing_machine', 'refrigerator', 'television', 'microwave',
    'kettle', 'computer', 'dishwasher', 'lighting'
]
```
​
### Adding rooms
1. Append the new room name to the rooms list in config/settings.py.
2. Add a corresponding entry to room_temperature in emulation/sensors/temperature.py.
3. Add a corresponding entry to room_humidity in emulation/sensors/humidity.py.
4. Add 24-hour probability arrays to both prob_weekday_occupancy and prob_weekend_occupancy in emulation/sensors/occupancy.py.

> Skipping any of the sensor files will raise a KeyError at emulation time because each module looks up every room by name.
​
### Adding appliances
1. Append the new appliance name to the appliances list in config/settings.py.
2. Add an entry to appliance_power in emulation/sensors/smart_plug.py with base, active, and always_on fields.
3. Add a schedule entry to appliance_schedules with default, weekday, and weekend keys.
4. Add a maximum power cap to appliance_max.
​
### Anomaly rates
Anomalies are injected into smart plug readings after the base power value is generated. The rates are read directly from config/settings.py.

| Setting	| Value	| Behaviour |
|--|--|--|
|anomaly_rate |	0.20	| Overall probability that any reading is anomalous |
|anomaly_rate_active	| 0.10	| When power > 10 W: random spike of 1.5×–3.0× normal power
|anomaly_rate_inactive	| 0.015	| When power ≤ 10 W: phantom load injected for appliances that have one (television, computer, lighting)|

> The anomaly_rate parameter in settings.py reflects the dataset-level target. The actual injection mechanism uses anomaly_rate_active and anomaly_rate_inactive separately inside apply_anomalies() in emulation/sensors/smart_plug.py.
​
### Sampling intervals
| Sensor	| Setting	| Default |
|--|--|--|
|Temperature |	temp_interval	| 30 s |
|Humidity	| hum_interval	| 30 s |
|Occupancy	| occ_interval	| 10 s |
|Smart plug |	sp_interval	| 5 s |


> The intervals control how often each sensor module publishes a reading to the MQTT broker during live emulation. They do not affect the sample_freq used to build the offline CSV dataset.

## Machine Learning model hyperparameter configuration
Configure Linear Regression, Random Forest, and MLP Regressor hyperparameter grids, GridSearchCV folds, and evaluation metric targets.

The pipeline trains three regression models to predict household energy consumption. Model definitions and their hyperparameter search grids live in models/train.py inside the get_models_and_params() function. Linear Regression is trained directly; Random Forest and MLP Regressor are tuned with GridSearchCV using the fold count defined in config/settings.py.
​
### Models
* Linear Regression
* Random Forest Regressor
* MLP Regressor

​
Full get_models_and_params() function
```python
models/train.py
def get_models_and_params():
    return {
        'LinearRegression': {
            'model': LinearRegression(),
            'params': {}
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=r_seed, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 500],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'MLPRegressor': {
            'model': MLPRegressor(random_state=r_seed, max_iter=500, early_stopping=True,
                                  validation_fraction=0.1, n_iter_no_change=20),
            'params': {
                'hidden_layer_sizes': [(64,32), (128,64), (128,64,32)],
                'activation': ['relu', 'tanh'],
                'learning_rate_init': [0.001, 0.01]
            }
        }
    }
```
​
### Evaluation targets
After training, the evaluation phase measures each model against the thresholds defined in config/settings.py. A model passes only if it meets all three metric targets.

| Metric	| Setting	| Target |
|--|--|--|
| R² |	r2_target	| ≥ 0.70 |
| RMSE	| rmse_target	| ≤ 100 W |
|MAE	| mae_target	| ≤ 70 W |
​
### Statistical analysis settings
| Setting	| Value	| Purpose |
|--|--|--|
| significance	| 0.05	| p-value threshold for hypothesis tests on model differences |
| permutation_repeats	| 50	| Number of shuffles per feature when computing permutation importance |
| validation_tolerance	| 0.20	| Allowed ±20 % deviation in cross-validation checks

> folds = 5 in config/settings.py controls the cv argument passed to every GridSearchCV call. Increasing it improves estimate stability but multiplies total training runs proportionally — for example, changing to 10 folds doubles the 135 Random Forest fits to 270.

> Expanding the hyperparameter grids without narrowing other axes can make training time grow quickly. Profile on a data subset before running a wide search on the full two-year dataset.
​
### Training run summary
| Model	| Combinations	| Folds	| Total fits |
|--|--|--|--|
| Linear Regression	| —	| —	| 1 |
| Random Forest	| 27	| 5	| 135 |
| MLP Regressor	| 12	| 5 | 60 |

## Emulation Module: generate_dataset and sensors
API reference for the emulation module — generate_dataset, mqtt_publisher, sensor_tests, and sensor submodules: temperature, humidity, occupancy, smart plug.

The emulation module generates synthetic smart home sensor data. It covers two modes: direct dataset generation (writing a CSV in one pass) and MQTT-based streaming (publishing readings to a broker tick by tick). Four sensor submodules — temperature, humidity, occupancy, and smart plug — provide the underlying data-generation logic used by both modes.
​
### generate_dataset
emulation/generate_dataset.py iterates over every simulated time interval, calls each sensor submodule, and assembles a single wide-format pandas.DataFrame.
>generate_dataset()

​
### mqtt_publisher
emulation/mqtt_publisher.py connects to an MQTT broker and streams sensor readings either as fast as possible (simulated) or at real sensor intervals (realtime).

>create_mqtt_client()
>run_emulation(client, mode='simulated')

​
### sensor_tests
emulation/sensor_tests.py validates the quality and realism of a generated dataset. All functions accept a pandas.DataFrame in the format returned by generate_dataset.
>test_summary_stats(df)
>test_time_patterns(df)
>test_weekday_weekend(df)
>test_all_appliances(df)
>test_correlations(df)
>test_full_timeseries(df)

​
### Sensor submodules
Each sensor submodule exposes one or two public functions. They are called by both generate_dataset and run_emulation.
​
**Temperature — emulation/sensors/temperature.py**
>generate_temperature(room, hour, day_of_week, noise=0.5)

**Humidity — emulation/sensors/humidity.py**
>generate_humidity(room, hour, temperature, noise=2.0)

**Occupancy — emulation/sensors/occupancy.py**
>generate_occupancy(room, hour, is_weekend)

​**Smart plug — emulation/sensors/smart_plug.py**
>generate_power(appliance, hour, occupancy, weekend=False, noise=5.0)
>apply_anomalies(appliance, power)

## Pipeline Module: Preprocessing and Splits
API reference for pipeline — run_preprocessing, prepare_data, save_split, load_split, and the MQTT subscriber that writes sensor data to daily CSVs.

The pipeline module handles everything between raw sensor data and model-ready arrays. preprocess.py detects the raw data format, loads and cleans the CSV files, pivots long-format MQTT data to wide format, and engineers the eight features used for modelling. split.py performs a temporal train/test split and scales features with StandardScaler. subscriber.py is a long-running MQTT subscriber that writes incoming messages to daily CSV files.
​
### preprocess
**pipeline/preprocess.py**
>detect_format(filepath)
>load_wide_format(raw_dir)
>load_long_format(raw_dir)
>wide_format(df, resample_freq)
>engineer_features(df_wide)
>run_preprocessing(raw_dir, resample_freq, save)

​
### split
**pipeline/split.py**
>prepare_data(features, test_ratio)
>save_split(X_train, y_train, X_test, y_test, scaler, feature_cols, output_dir)
>load_split(input_dir)

​
### subscriber
pipeline/subscriber.py is a standalone MQTT subscriber. It subscribes to all topics under {mqtt_topic_prefix}/#, parses incoming JSON payloads, and writes each reading as a row in a daily CSV file under data/raw/. A background thread logs message counts, error rates, and throughput every 30 seconds.

**CSV format written:** timestamp, room, sensor_type, appliance, value

Each calendar day gets its own file (e.g., data/raw/2023-01-01.csv). File handles are managed thread-safely with a write lock.
**Usage:**
```python
# Run the subscriber as a background process alongside the publisher
python -m pipeline.subscriber
```
```python
# Launch programmatically (e.g., from run_emulation.py)
from pipeline.subscriber import main
main()
```
>The subscriber handles graceful shutdown on SIGINT (Ctrl+C) and SIGTERM, flushing all open file handles before disconnecting.

## Models Module: training and evaluation
API reference for the models module — train_all_models, evaluate_all_models, compare_models, and diagnostic plot functions for energy prediction.

## Analysis Module: importance and validation
API reference for the analysis module — permutation importance, Oh decomposition, and run_validation for interpreting and validating prediction models.

The analysis module provides three interpretability and validation tools. feature_importance.py measures how much each feature contributes to model R² using permutation importance with statistical significance testing. decomposition.py decomposes each feature’s importance into an independent component and an interaction component following the Oh (2022) method. validation.py compares emulated sensor distributions against two real-world datasets using the Kolmogorov–Smirnov test and a percentage-difference tolerance check.
​
### feature_importance
**analysis/feature_importance.py**
>calc_permutation_importance(model, X_test, y_test, feature_names, model_name)
>run_permutation_importance(trained_res, X_test, y_test, feature_names)
>save_importance_data(all_importance, output_path)
>plot_importance_bars(all_importance, output_path)
>plot_importance_heatmap(all_importance, output_path)

​
### decomposition
analysis/decomposition.py — Oh (2022) feature power interaction decomposition.
For each feature, the method measures:
* Feature power — the model’s independent reliance on that feature (all other features permuted).
* Interaction power — the additional contribution that comes from correlations with other features (total_importance − feature_power, floored at zero).
* Total importance — the overall MAE increase when only that feature is permuted.
>run_oh_decomposition(trained_res, X_test, y_test, feature_names, repeats)
>save_decomposition_csv(decomposition, output_path)
>plot_decomposition_bars(decomposition, output_path)
>plot_decomposition_stacked(decomposition, output_path)

​
### validation
analysis/validation.py — comparison of emulated distributions against real-world datasets.
>run_validation(emulated_features_path=None)

## Outputs
### Data formats: raw and processed CSV schemas
Reference for CSV schemas in Smart Home Energy Prediction — wide-format raw sensor data, long-format MQTT output, and the engineered features dataset.

Every stage of the pipeline reads or writes CSVs in a well-defined schema. This page documents the exact column layout of each file so you can inspect, debug, or extend the data at any point.
​
#### data/raw/
Raw sensor data written by the emulation phases lives here. Files are named by the phase that generated them. The preprocessing pipeline loads all *.csv files from this directory and concatenates them before any processing takes place.
​
#### Wide format (Phase 1 — direct generation)
Phase 1 (emulation/generate_dataset.py) writes one row per 5-minute interval. Each sensor reading gets its own named column, giving the file its “wide” shape.
```python
timestamp,hour,day_of_week,is_weekend,temperature_bedroom,temperature_living_room,temperature_bathroom,temperature_kitchen,temperature_studio,humidity_bedroom,humidity_living_room,humidity_bathroom,humidity_kitchen,humidity_studio,occupancy_bedroom,occupancy_living_room,occupancy_bathroom,occupancy_kitchen,occupancy_studio,power_washing_machine,power_refrigerator,power_television,power_microwave,power_kettle,power_computer,power_dishwasher,power_lighting,mean_temperature,mean_humidity,occupancy_count,active_appliances,total_power
2023-01-01 00:00:00,0,6,1,19.2,18.7,17.1,16.8,18.3,52.1,50.4,61.3,55.8,49.7,0,0,0,0,0,0.0,42.5,0.0,0.0,0.0,0.0,0.0,15.3,18.02,53.86,0,2,57.8
2023-01-01 00:05:00,0,6,1,19.1,18.6,17.0,16.9,18.4,52.3,50.6,61.1,55.9,49.5,0,0,0,0,0,0.0,42.5,0.0,0.0,0.0,0.0,0.0,15.3,18.00,53.88,0,2,57.8
```
>Wide-format files may also have an Unnamed: 0 column if they were written with a default pandas index. The preprocessor detects and handles this automatically, using it as the timestamp index.

#### Wide format column reference
| Column	| Type	| Description |
|--|--|--|
| timestamp	| datetime	| UTC timestamp at 5-minute intervals |
| hour	| int	| Hour of day (0–23) |
| day_of_week	| int	| Day of week (0 = Monday, 6 = Sunday) |
| is_weekend	| int	| 1 if Saturday or Sunday, else 0 |
| temperature_<room> |	float	| Room temperature in °C |
| humidity_<room>	| float	| Relative humidity (%) |
| occupancy_<room>	| int	| Room occupancy (0 or 1) |
| power_<appliance>	| float	| Appliance power draw in watts |
| mean_temperature	| float	| Mean temperature across all rooms |
| mean_humidity	| float	| Mean humidity across all rooms |
| occupancy_count	| int	| Number of rooms currently occupied |
| active_appliances	| int	| Number of appliances drawing >10 W |
| total_power	| float	| Sum of all appliance power in watts |

Rooms: **bedroom, living_room, bathroom, kitchen, studio**
Appliances: **washing_machine, refrigerator, television, microwave, kettle, computer, dishwasher, lighting**
​
#### Long format (Phase 2 — MQTT)
Phase 2 emits one measurement per row. This is the format written by the MQTT subscriber and is significantly more rows than the equivalent wide-format file.
```python
timestamp,room,sensor_type,appliance,value
2023-01-01 00:00:00,bedroom,temperature,,19.2
2023-01-01 00:00:00,living_room,humidity,,50.4
2023-01-01 00:00:00,bedroom,occupancy,,0
2023-01-01 00:00:00,,power,washing_machine,0.0
2023-01-01 00:00:00,,power,refrigerator,42.5
```

#### Long format column reference
| Column	| Type	| Description |
|--|--|--|
| timestamp	| datetime	| Measurement timestamp |
| room	| string	| Room name; empty for appliance rows |
| sensor_type	| string	| temperature, humidity, occupancy, or power |
| appliance	| string	| Appliance name; empty for environmental rows |
| value	| float	| Sensor reading in native units |
​
#### Format detection
detect_format() in pipeline/preprocess.py reads the first three rows of the first CSV file and inspects the column names:
```python
def detect_format(filepath):
    df_sample = pd.read_csv(filepath, nrows=3)
    cols = df_sample.columns.tolist()

    has_sensor_cols = any(c.startswith('temperature_') for c in cols)
    has_power_cols = any(c.startswith('power_') for c in cols)

    if has_sensor_cols and has_power_cols:
        return 'wide'

    has_long_cols = ('sensor_type' in cols or 'room' in cols)
    if has_long_cols:
        return 'long'

    return 'wide'  # default
```
>Wide format detection
>Long format detection
>Default fallback

​#### data/processed/
After preprocessing, two CSV files and four NumPy split files are written to data/processed/.
​
#### wide_data.csv

Produced by run_preprocessing(). If the raw input was long format, this file is the result of pivoting and resampling to 5-minute intervals. If the raw input was already wide, this file is the cleaned and concatenated version.
The schema mirrors the wide-format raw file (minus the pre-computed derived columns), with timestamp as the index.

```python
timestamp,temperature_bedroom,temperature_living_room,...,power_lighting
2023-01-01 00:00:00,19.2,18.7,...,15.3
```

#### features_data.csv
Produced by engineer_features(). Contains exactly eight engineered columns derived from wide_data.csv. This is the file consumed by the model training pipeline.

```python
timestamp,hour,day_of_week,is_weekend,mean_temperature,mean_humidity,occupancy_count,active_appliances,total_power
2023-01-01 00:00:00,0,6,1,18.02,53.86,0,2,57.8
2023-01-01 00:05:00,0,6,1,18.00,53.88,0,2,57.8
```

#### Engineered feature column reference
| Column	| Type	| Description |
|--|--|--|
| hour	| int	| Hour extracted from the timestamp index (0–23) |
| day_of_week	| int	| Day of week from the timestamp index (0–6) |
| is_weekend	| int	| 1 if day_of_week ≥ 5, else 0 |
| mean_temperature	| float	| Mean of all temperature_<room> columns, rounded to 2 dp |
| mean_humidity	| float	| Mean of all humidity_<room> columns, rounded to 2 dp |
| occupancy_count	| int	| Sum of all occupancy_<room> columns |
| active_appliances	| int	| Count of power_<appliance> columns where value > 10 W |
| total_power	| float	| Sum of all power_<appliance> columns, rounded to 1 dp; this is the model target |

>total_power is both an engineered feature in features_data.csv and the target variable used during training. The split step separates it from the feature matrix automatically.
​
#### Split files

save_split() in pipeline/split.py writes six files after the temporal train/test split (80% train, 20% test):

```python
np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/y_test.npy', y_test)

with open('data/processed/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('data/processed/feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
```

| File	| Format	| Contents |
|--|--|--|
| X_train.npy	| NumPy array	| Scaled feature matrix for the training set |
| X_test.npy	| NumPy array	| Scaled feature matrix for the test set |
| y_train.npy	| NumPy array	| total_power target values for the training set |
| y_test.npy	| NumPy array	| total_power target values for the test set |
| scaler.pkl	| pickle	| Fitted StandardScaler instance |
| feature_cols.pkl	| pickle	| Ordered list of feature column names |

>The split is temporal: the test set is always the most recent 20% of rows. This preserves time-series integrity and ensures models are evaluated on future data rather than randomly sampled points.
​
#### data/validation/
This directory is the input location for real-world datasets used in Phase 6 validation. The pipeline looks for two specific filenames:
* UCI Appliances Energy: Place energydata_complete.csv here. Download from the UCI Machine Learning Repository (dataset ID 374).
* Kaggle Smart Home: Place HomeC.csv here. Download from the Kaggle Smart Home Dataset with Weather Information.

If neither file is present when analysis/validation.py runs, the module logs a warning and returns an empty results list without raising an error. Validation outputs are written to results/ alongside model results.

### Results directory: outputs, metrics, and plots
All files written to results/ — trained model pickles, evaluation CSVs, feature importance data, decomposition outputs, and generated plot descriptions.

Every artifact produced by training, evaluation, and analysis is written to the results/ directory. This page lists each file, its format, and the exact columns or content it contains so you can load and interpret outputs without digging into the source code.
​
#### Trained models
models/train.py writes model artifacts to results/ after training completes.
* **trained_models.pkl**: Full results dictionary keyed by model name. Each entry holds the estimator, best hyperparameters, CV score, CV results, and training time.
* **Individual model files**: One pickle per model: LinearRegression_model.pkl, RandomForest_model.pkl, MLPRegressor_model.pkl. Each contains only the fitted estimator.

#### Loading trained models

Use load_models() to reload the full results dictionary:

```python
from models.train import load_models

results = load_models('results')
# results is a dict with keys: 'LinearRegression', 'RandomForest', 'MLPRegressor'
# Each value has: estimator, best_params, cv_score, cv_results, training_time

estimator = results['RandomForest']['estimator']
y_pred = estimator.predict(X_test)
```

To load a single estimator directly:

```python
import pickle

with open('results/RandomForest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
```

#### Evaluation outputs
models/evaluate.py writes two CSV files and three plots after running evaluate_all_models().
​
#### evaluation_metrics.csv
One row per model with performance metrics and pass/fail flags against the dissertation targets.

```python
model_name,r2,rmse,mae,meets_r2_target,meets_rmse_target,meets_mae_target,best_params,training_time
LinearRegression,0.7821,87.43,61.20,True,True,True,None,0.1
RandomForest,0.9134,55.67,38.41,True,True,True,"{'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}",142.3
MLPRegressor,0.8902,63.15,44.88,True,True,True,"{'activation': 'relu', 'hidden_layer_sizes': (128, 64), 'learning_rate_init': 0.001}",98.7
```

| Column	| Type	| Description |
|--|--|--|
| model_name	| string	| Model identifier (LinearRegression, RandomForest, MLPRegressor) |
| r2	| float	| R² coefficient of determination on the test set (4 dp) |
| rmse	| float	| Root mean squared error in watts (2 dp) |
| mae	| float	| Mean absolute error in watts (2 dp) |
| meets_r2_target	| bool	| True if R² ≥ 0.70 |
| meets_rmse_target	| bool	| True if RMSE ≤ 100 W |
| meets_mae_target	| bool	| True if MAE ≤ 70 W |
| best_params	| string	| Stringified dict of best hyperparameters from GridSearchCV; None for LinearRegression|
| training_time	| float	| Wall-clock training time in seconds (1 dp) |
​
#### model_comparisons.csv

All pairwise paired t-tests on absolute prediction errors. With three models, this file always has three rows.

```python
Model_A,Model_B,MAE_A,MAE_B,t_statistic,p_value,significant,better_model
LinearRegression,RandomForest,61.20,38.41,18.431,0.0000,True,RandomForest
LinearRegression,MLPRegressor,61.20,44.88,12.765,0.0000,True,MLPRegressor
RandomForest,MLPRegressor,38.41,44.88,-6.342,0.0000,True,RandomForest
```

| Column	| Type	| Description |
|--|--|--|
| Model_A	| string	| First model in the comparison pair |
| Model_B	| string	| Second model in the comparison pair |
| MAE_A	| float	| Mean absolute error of Model A on the test set |
| MAE_B	| float	| Mean absolute error of Model B on the test set |
| t_statistic	| float	| Paired t-test statistic |
| p_value	| float	| Two-tailed p-value |
| significant	| bool	| True if p-value < 0.05 |
| better_model	| string	| Name of the model with lower mean absolute error |
​
#### Performance targets
The pass/fail flags in evaluation_metrics.csv are evaluated against these thresholds, defined in config/settings.py:

| Metric	| Target |
|--|--|
| R²	| ≥ 0.70 |
| RMSE	| ≤ 100 W |
| MAE	| ≤ 70 W |

>The evaluate.py module docstring cites an MAE target of 120 W, but config/settings.py sets mae_target = 70. The value from settings.py is what the pipeline enforces at runtime.
​
#### Evaluation plots
>prediction_vs_actual.png
>residual_analysis.png
>model_comparison_bars.png

​#### Feature importance outputs
analysis/feature_importance.py runs permutation importance after model evaluation and writes the following files.
​
#### permutation_importance.csv
One row per (model, feature) pair. Features are sorted by descending mean_importance within each model.

```python
model,feature,rank,mean_importance,std_importance,t_stat,p_value,significant
RandomForest,total_power,1,0.412300,0.018200,16.0321,0.000001,True
RandomForest,active_appliances,2,0.187400,0.012100,10.9412,0.000003,True
RandomForest,hour,3,0.091200,0.009300,6.9241,0.000120,True
```

| Column	| Type	| Description |
|--|--|--|
| model	| string	| Model name |
| feature	| string	| Feature name from feature_cols |
| rank	| int	| Importance rank within the model (1 = most important) |
| mean_importance	| float	| Mean R² decrease across permutation repeats (6 dp) |
| std_importance	| float	| Standard deviation of R² decrease across repeats (6 dp) |
| t_stat	| float	| One-sample t-statistic: mean / (std / √repeats) (4 dp) |
| p_value	| float	| One-tailed p-value from the t-distribution (6 dp) |
| significant	| bool	| True if p-value < 0.05 |
​
#### permutation_importance.png
One horizontal bar chart per model. Bars show mean importance with error bars (±1 std). Bars are coloured red if the feature is statistically significant at α = 0.05, blue otherwise. Features are ordered from most to least important (top to bottom).
Saved at 300 dpi with bbox_inches='tight'.
​
#### importance_heatmap.png
A heatmap with features on the y-axis and models on the x-axis. Cell colour uses a coolwarm scale indicating R² decrease. Each cell is annotated with its numeric value. This makes it easy to spot features that matter across all models versus those that are model-specific.
Saved at 150 dpi.
​
### Decomposition outputs
analysis/decomposition.py runs Oh (2022) feature power interaction decomposition and writes three files.
​
#### oh_decomposition.csv
One row per (model, feature) pair, sorted by descending total_importance_mean within each model.
```python
model,feature,total_importance_mean,total_importance_std,feature_power_mean,feature_power_std,interaction_power_mean,interaction_power_std,feature_power_ratio,base_mae
RandomForest,active_appliances,0.124300,0.018200,0.098400,0.012100,0.025900,0.009300,0.7917,38.41
```
| Column	| Type	| Description |
|--|--|--|
| model	| string	| Model name |
| feature	| string	| Feature name |
| total_importance_mean	| float	| Mean total MAE contribution (feature power + interaction power)|
| total_importance_std	| float	| Std of total importance across repeats |
| feature_power_mean	| float	| Mean direct (independent) contribution to MAE |
| feature_power_std	| float	| Std of feature power across repeats |
| interaction_power_mean	| float	| Mean contribution through interactions with other features |
| interaction_power_std	| float	| Std of interaction power across repeats |
| feature_power_ratio	| float	| feature_power_mean / total_importance_mean — proportion of importance that is independent |
| base_mae	| float	| Baseline MAE of the model before any permutation |
​
#### oh_decomposition.png
One horizontal grouped bar chart per model. Each feature gets two side-by-side bars: feature power (blue) and interaction power (orange). Features are sorted by total importance.
​
#### oh_decomposition_stacked.png
A single stacked horizontal bar chart for the first model in the results dictionary. Each bar shows the feature power segment (blue) and interaction power segment (orange) stacked. The percentage of importance that is independent is annotated to the right of each bar.
​
#### Validation outputs
analysis/validation.py writes these files when real-world datasets are present in data/validation/.
>validation_results.csv
>validation_summary.png
>distribution_comparison_<dataset>.png

>If data/validation/ contains no recognised dataset files, the validation module exits gracefully with a logged warning. All other results/ outputs are unaffected.
