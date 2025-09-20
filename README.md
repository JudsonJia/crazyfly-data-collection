# Crazyflie Data Collection System

A comprehensive data collection framework for autonomous drone trajectory recording using Crazyflie 2.1. This system captures high-frequency flight data including position, velocity, attitude, and acceleration for machine learning research.

The following pictures showing our data collection environment:
<p align="center">
  <img src="https://github.com/user-attachments/assets/871da981-ce22-471c-940e-c4fd14b1b611" width="45%" />
  <img src="https://github.com/user-attachments/assets/8ad24636-8b4b-4509-ab20-f7ab40dd7a1e" width="45%" />
</p>

## Features

- **Real-time Data Collection**: Continuous logging at 100Hz with time-synchronized multi-sensor fusion
- **Time Slot Synchronization**: Advanced 10ms time slot mechanism for precise data alignment
- **Multi-Pattern Support**: Built-in flight patterns (vertical, square, triangle trajectories)
- **Custom Trajectory**: Support for user-defined waypoint sequences
- **Battery Safety**: Automatic voltage monitoring and safety checks

## Hardware Requirements

- Crazyflie 2.1 drone with flow deck v2
- Crazyradio PA USB dongle
- LPS positioning system
- Battery voltage > 3.8V for safe operation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/crazyflie-data-collection.git
cd crazyflie-data-collection

# Install dependencies
pip install cflib numpy matplotlib

# Verify Crazyradio connectivity
python ai_data_collector.py --test-only
```

## Quick Start

### Basic Trajectory Collection

```bash
# Collect vertical flight data
python ai_data_collector.py --pattern vertical_med

# Collect square trajectory data
python ai_data_collector.py --pattern square_large

# Collect triangle trajectory data
python ai_data_collector.py --pattern triangle_small
```

### Custom Trajectories

```bash
# Define custom waypoints (x,y,z coordinates)
python ai_data_collector.py --custom "0,0,0.3;0.5,0,0.6;0.5,0.5,0.6;0,0,0.3"
```

### Advanced Options

```bash
# Use specific URI
python ai_data_collector.py --uri radio://0/80/2M/E7E7E7E7E6 --pattern square_med

# Skip battery check (use with caution)
python ai_data_collector.py --pattern vertical_large --disable-battery

# Test connection only
python ai_data_collector.py --test-only
```

## Available Flight Patterns

| Pattern | Waypoints | Description |
|---------|-----------|-------------|
| `vertical_small` | 5 points | Multi-level vertical flight (0.3-0.9m) |
| `vertical_med` | 3 points | Medium vertical flight (0.4-0.8m) |
| `vertical_large` | 3 points | Large vertical flight (0.5-1.0m) |
| `square_small` | 5 points | 0.3m × 0.3m square at 0.3m height |
| `square_med` | 5 points | 0.5m × 0.5m square at 0.4m height |
| `square_large` | 5 points | 0.7m × 0.7m square at 0.5m height |
| `triangle_small` | 4 points | 0.4m triangle at 0.6m height |
| `triangle_med` | 4 points | 0.5m triangle at 0.6m height |
| `triangle_large` | 4 points | 0.8m triangle at 0.6m height |

## Data Output Format

The system generates JSON files in the following structure:

```json
{
  "mission": {
    "name": "square_med",
    "timestamp": "20250120_143022",
    "waypoints": [[0,0,0.4], [0.5,0,0.4], ...],
    "duration": 12.5,
    "samples": 1250
  },
  "data": {
    "states": [
      {
        "time": 0.125,
        "vector": [x, y, z, vx, vy, vz, roll, pitch, yaw, ax, ay, az],
        "components": {
          "pos": [x, y, z],
          "vel": [vx, vy, vz],
          "att": [roll, pitch, yaw],
          "acc": [ax, ay, az]
        }
      }
    ],
    "commands": [
      {
        "timestamp": 0.120,
        "target": [x, y, z, yaw]
      }
    ]
  },
  "stats": {
    "collection": {
      "created": 1280,
      "complete": 1250,
      "merged": 15,
      "dropped": 15
    },
    "flight": {
      "max_vel": 0.85,
      "mean_vel": 0.23,
      "envelope": {
        "x": [-0.02, 0.52],
        "y": [-0.01, 0.51],
        "z": [0.38, 0.42]
      }
    }
  }
}
```

## System Architecture

### Core Components

1. **PureContinuousCollector**: Time-synchronized data collection with 10ms time slots
2. **PositionManager**: Real-time position tracking and validation
3. **Flight Controller**: Precision waypoint navigation with 100Hz control frequency
4. **Data Synchronization**: Advanced time slot mechanism for multi-sensor alignment

### Data Collection Process

1. **Initialization**: Kalman filter reset and parameter optimization
2. **Pre-flight**: Battery check and position estimator validation
3. **Mission Execution**: Waypoint navigation with continuous data logging
4. **Data Processing**: Time slot synchronization and quality validation
5. **Output Generation**: Structured JSON format with metadata

## Delay Testing and Optimization

Use the delay tester to optimize data collection parameters:

```bash
# Test LogConfig timing
python crazyflie_delay_tester.py --duration 30

# Verbose output for debugging
python crazyflie_delay_tester.py --duration 60 --verbose
```

The delay tester provides recommendations for optimal time slot configuration based on your system's performance.

## Configuration

### Key Parameters

- `SLOT_SIZE_MS = 10`: Time slot size for data synchronization
- `TOLERANCE_MS = 5`: Tolerance for time slot merging
- `BATTERY_MIN_VOLTAGE = 3.8`: Minimum safe battery voltage
- Control frequency: 100Hz for stable flight control


## Related Work

- [Drone Trajectory LSTM](https://github.com/yourusername/drone-trajectory-lstm) - Model training using data from this collection system

