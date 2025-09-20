"""
Crazyflie Data Collection script

Usage method:
# Basic trajectory collection
python ai_data_collector.py --pattern vertical_small
python ai_data_collector.py --pattern vertical_med
python ai_data_collector.py --pattern vertical_large
python ai_data_collector.py --pattern square_small
python ai_data_collector.py --pattern square_med
python ai_data_collector.py --pattern square_large
python ai_data_collector.py --pattern triangle_small
python ai_data_collector.py --pattern triangle_med
python ai_data_collector.py --pattern triangle_large

# Custom trajectory (format: x,y,z;x,y,z;...)
python ai_data_collector.py --custom "0,0,0.3;0,0,0.8;0,0,0.3"
python ai_data_collector.py --custom "0,0,0.6;0.5,0,0.6;0.5,0.5,0.6;0,0,0.6"

# Connection and system tests
python ai_data_collector.py --test-only
python ai_data_collector.py --test-only --disable-battery

# Advanced options
python ai_data_collector.py --pattern vertical_med --disable-battery
python ai_data_collector.py --uri radio://0/80/2M/E7E7E7E7E6 --pattern square_large

Note: Default URI is radio://0/80/2M/E7E7E7E7E6 if not specified
     --disable-battery skips battery voltage check (use with caution)
     --test-only performs connection test without flying
"""
import math
import time
import logging
import argparse
import os
import json
import datetime
import threading
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper
from cflib.utils.reset_estimator import reset_estimator

# ========================= Configuration =========================
os.makedirs('training_data', exist_ok=True)
os.makedirs('logs', exist_ok=True)

BATTERY_MIN_VOLTAGE = 3.8
SLOT_SIZE_MS = 10
TOLERANCE_MS = 5

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/collector_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('collector')

# ========================= Data structure =========================
@dataclass
class DroneState:
    """12-dimensional state vector"""
    timestamp: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.vx, self.vy, self.vz,
                self.roll, self.pitch, self.yaw, self.ax, self.ay, self.az]

# ========================= Position Manager =========================
class PositionManager:

    def __init__(self, scf):
        self.scf = scf
        self.position = {'x': 0, 'y': 0, 'z': 0, 'yaw': 0}
        self.position_lock = threading.Lock()
        self.running = False
        self.log_config = None
        self.last_update_time = 0

    def start(self):
        if not self.running:
            try:
                self.log_config = LogConfig('PositionManager', period_in_ms=50)
                self.log_config.add_variable('kalman.stateX', 'float')
                self.log_config.add_variable('kalman.stateY', 'float')
                self.log_config.add_variable('kalman.stateZ', 'float')
                self.log_config.add_variable('stabilizer.yaw', 'float')

                self.log_config.data_received_cb.add_callback(self._position_callback)

                self.scf.cf.log.add_config(self.log_config)
                self.log_config.start()
                self.running = True
                logger.info("Position manager started")

                # Wait for first position update
                timeout = 5.0
                start_wait = time.time()
                while self.last_update_time == 0 and time.time() - start_wait < timeout:
                    time.sleep(0.1)

                if self.last_update_time == 0:
                    logger.warning("Position manager startup timeout")
                else:
                    logger.info("Position manager data flow is normal")

            except Exception as e:
                logger.error(f"Position manager startup failed: {e}")
                self.running = False

    def stop(self):
        """Stop position updates"""
        if self.running and self.log_config:
            try:
                self.log_config.stop()
                time.sleep(0.1)  # Make sure callback completes
                logger.info("Position manager stopped")
            except Exception as e:
                logger.debug(f"Error when stopping position manager: {e}")
            finally:
                self.running = False
                self.log_config = None

    def _position_callback(self, timestamp, data, logconf):
        """Position callback"""
        try:
            with self.position_lock:
                self.position.update({
                    'x': data['kalman.stateX'],
                    'y': data['kalman.stateY'],
                    'z': data['kalman.stateZ'],
                    'yaw': data['stabilizer.yaw']
                })
                self.last_update_time = time.time()
        except Exception as e:
            logger.debug(f"Position callback error: {e}")

    def get_position(self) -> Dict[str, float]:
        """Get current position"""
        with self.position_lock:
            return self.position.copy()

    def is_position_valid(self) -> bool:
        """Check if position data is valid"""
        return self.last_update_time > 0 and time.time() - self.last_update_time < 0.5

# ========================= Core data collector =========================
class PureContinuousCollector:
    """Continuous data collector"""

    def __init__(self, slot_size_ms: int = SLOT_SIZE_MS):
        self.slot_size_ms = slot_size_ms
        self.tolerance_ms = TOLERANCE_MS

        # Time slots
        self.time_slots = {}
        self.slot_lock = threading.Lock()

        # Collection status
        self.collecting = False
        self.start_time = None

        # Output data
        self.states = []
        self.commands = []
        self.data_lock = threading.Lock()

        # Statistics
        self.stats = {'created': 0, 'complete': 0, 'merged': 0, 'dropped': 0}

        self.log_configs = []

    def get_slot_id(self, cf_timestamp_us: float) -> int:
        """CF timestamp → slot ID"""
        ms = cf_timestamp_us / 1000
        return int(ms / self.slot_size_ms) * self.slot_size_ms

    def start(self, scf):
        """Begin collection"""
        self.collecting = True
        self.start_time = time.time()
        self.time_slots.clear()
        self.states.clear()
        self.commands.clear()

        logger.info(f"Start collection (slot:{self.slot_size_ms}ms)")

        # Position-velocity group
        self._start_pos_vel_log(scf)
        # Attitude-IMU group
        self._start_att_imu_log(scf)
        # Cleanup thread
        self._start_cleanup()

    def _start_pos_vel_log(self, scf):
        """Position-velocity LogConfig"""
        config = LogConfig('PosVel', period_in_ms=self.slot_size_ms)
        config.add_variable('kalman.stateX', 'float')
        config.add_variable('kalman.stateY', 'float')
        config.add_variable('kalman.stateZ', 'float')
        config.add_variable('kalman.statePX', 'float')
        config.add_variable('kalman.statePY', 'float')
        config.add_variable('kalman.statePZ', 'float')

        config.data_received_cb.add_callback(
            lambda ts, data, conf: self._data_callback(ts, data, 'pos_vel')
        )

        scf.cf.log.add_config(config)
        config.start()
        self.log_configs.append(config)

    def _start_att_imu_log(self, scf):
        """Attitude-IMU LogConfig"""
        config = LogConfig('AttIMU', period_in_ms=self.slot_size_ms)
        config.add_variable('stabilizer.roll', 'float')
        config.add_variable('stabilizer.pitch', 'float')
        config.add_variable('stabilizer.yaw', 'float')
        config.add_variable('acc.x', 'float')
        config.add_variable('acc.y', 'float')
        config.add_variable('acc.z', 'float')

        config.data_received_cb.add_callback(
            lambda ts, data, conf: self._data_callback(ts, data, 'att_imu')
        )

        scf.cf.log.add_config(config)
        config.start()
        self.log_configs.append(config)

    def _data_callback(self, cf_timestamp: float, data: Dict, group: str):
        """Data callback"""
        if not self.collecting:
            return

        with self.slot_lock:
            slot_id = self.get_slot_id(cf_timestamp)

            # Initialize slot
            if slot_id not in self.time_slots:
                self.time_slots[slot_id] = {}
                self.stats['created'] += 1

            # Store data
            self.time_slots[slot_id][group] = {
                'data': data,
                'rel_time': time.time() - self.start_time
            }

            # Check completeness
            if self._is_complete(slot_id):
                self._save_state(slot_id)
            else:
                # Try to merge with nearby slots
                self._try_merge(slot_id, group, data, cf_timestamp)

    def _is_complete(self, slot_id: int) -> bool:
        """Check slot completeness"""
        if slot_id not in self.time_slots:
            return False
        return {'pos_vel', 'att_imu'} <= set(self.time_slots[slot_id].keys())

    def _save_state(self, slot_id: int):
        """Save complete state"""
        slot = self.time_slots[slot_id]
        pos_data = slot['pos_vel']['data']
        att_data = slot['att_imu']['data']

        # Average time
        avg_time = (slot['pos_vel']['rel_time'] + slot['att_imu']['rel_time']) / 2

        state = DroneState(
            timestamp=avg_time,
            x=pos_data['kalman.stateX'],
            y=pos_data['kalman.stateY'],
            z=pos_data['kalman.stateZ'],
            vx=pos_data['kalman.statePX'],
            vy=pos_data['kalman.statePY'],
            vz=pos_data['kalman.statePZ'],
            roll=att_data['stabilizer.roll'],
            pitch=att_data['stabilizer.pitch'],
            yaw=att_data['stabilizer.yaw'],
            ax=att_data['acc.x'],
            ay=att_data['acc.y'],
            az=att_data['acc.z']
        )

        with self.data_lock:
            self.states.append(state)

        self.stats['complete'] += 1
        del self.time_slots[slot_id]

    def _try_merge(self, ideal_slot: int, group: str, data: Dict, cf_timestamp: float):
        """Try to merge with nearby slots"""
        tolerance_slots = self.tolerance_ms // self.slot_size_ms

        for offset in range(-tolerance_slots, tolerance_slots + 1):
            if offset == 0:
                continue

            candidate_slot = ideal_slot + (offset * self.slot_size_ms)

            if candidate_slot in self.time_slots:
                if group not in self.time_slots[candidate_slot]:
                    # Merge
                    self.time_slots[candidate_slot][group] = {
                        'data': data,
                        'rel_time': time.time() - self.start_time
                    }
                    self.stats['merged'] += 1

                    if self._is_complete(candidate_slot):
                        self._save_state(candidate_slot)
                    return

    def record_command(self, x: float, y: float, z: float, yaw: float):
        """Record control command"""
        if self.collecting:
            with self.data_lock:
                self.commands.append({
                    'timestamp': time.time() - self.start_time,
                    'target': [x, y, z, yaw]
                })

    def _start_cleanup(self):
        """Cleanup thread"""
        def worker():
            while self.collecting:
                current_time = time.time() - self.start_time
                cutoff = current_time - 0.2

                with self.slot_lock:
                    expired = []
                    for slot_id, slot_data in self.time_slots.items():
                        if any(g['rel_time'] < cutoff for g in slot_data.values()):
                            if not self._is_complete(slot_id):
                                expired.append(slot_id)

                    for slot_id in expired:
                        del self.time_slots[slot_id]
                        self.stats['dropped'] += 1

                time.sleep(1.0)

        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        """Stop collection"""
        self.collecting = False

        for config in self.log_configs:
            try:
                config.stop()
            except Exception as e:
                logger.debug(f"Error stopping LogConfig: {e}")
        self.log_configs.clear()

        # Process remaining complete slots
        with self.slot_lock:
            remaining = [s for s in self.time_slots.keys() if self._is_complete(s)]
            for slot_id in remaining:
                self._save_state(slot_id)

        logger.info(f"Collection complete - stats: {self.stats}")

    def get_data(self) -> Dict[str, Any]:
        """Get all data"""
        with self.data_lock:
            return {
                'states': sorted(self.states, key=lambda s: s.timestamp),
                'commands': sorted(self.commands, key=lambda c: c['timestamp']),
                'stats': self.stats.copy()
            }

# ========================= Battery check =========================
def check_battery_simple(scf) -> Tuple[float, bool]:
    """Battery check"""
    battery_log = LogConfig('BatteryCheck', period_in_ms=100)
    battery_log.add_variable('pm.vbat', 'float')

    try:
        with SyncLogger(scf, battery_log) as sync_logger:
            for log_entry in sync_logger:
                voltage = log_entry[1]['pm.vbat']
                is_safe = voltage >= BATTERY_MIN_VOLTAGE
                logger.info(f"Battery: {voltage:.2f}V ({'safe' if is_safe else 'low'})")
                return voltage, is_safe
    except Exception as e:
        logger.error(f"Battery check failed: {e}")
        return 0.0, False

def wait_for_estimator_improved(scf):
    logger.info('Waiting for position estimator...')

    log_config = LogConfig('VarianceCheck', period_in_ms=100)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    try:
        with SyncLogger(scf, log_config) as sync_logger:
            history = {'x': [1000]*10, 'y': [1000]*10, 'z': [1000]*10}
            threshold = 0.005
            count = 0

            for log_entry in sync_logger:
                data = log_entry[1]

                for axis, var in [('x', 'kalman.varPX'), ('y', 'kalman.varPY'), ('z', 'kalman.varPZ')]:
                    history[axis].append(data[var])
                    if len(history[axis]) > 10:
                        history[axis].pop(0)

                if all(len(history[axis]) >= 10 and max(history[axis]) < threshold for axis in history):
                    logger.info('Position estimator ready')
                    break

                count += 1
                if count % 10 == 0:
                    x_var, y_var, z_var = data['kalman.varPX'], data['kalman.varPY'], data['kalman.varPZ']
                    logger.info(f'Waiting for estimator... X: {x_var:.4f}, Y: {y_var:.4f}, Z: {z_var:.4f}')

                # Wait max 15 seconds
                if count > 150:
                    logger.warning('Estimator wait timeout, continuing')
                    break

    except Exception as e:
        logger.warning(f"Estimator check failed, skipping: {e}")

# ========================= Optimize controller parameters =========================
def optimize_controller_for_stability(scf):
    """Optimize control parameters for enhanced stability"""
    try:
        # Adjust PID parameters for altitude and stability
        scf.cf.param.set_value('posCtlPid.zKp', 1.5)  # Reduce P gain, reduce oscillation
        scf.cf.param.set_value('posCtlPid.zKi', 0.6)  # Increase I gain, improve steady-state accuracy
        scf.cf.param.set_value('posCtlPid.zKd', 0.4)  # Increase D gain, suppress overshoot
        scf.cf.param.set_value('velCtlPid.vzKd', 0.5)  # Increase velocity control damping

        logger.info("Controller parameters optimized")
    except Exception as e:
        logger.error(f"Failed to set controller parameters: {e}")

# ========================= Flight control =========================
def fly_to_point_fixed(scf, tx: float, ty: float, tz: float, tyaw: float,
                      collector: PureContinuousCollector, pos_manager: PositionManager,
                      timeout: float = 10.0) -> bool:
    """Flight control"""
    cf = scf.cf
    tolerance = 0.08
    stable_steps = 30

    logger.info(f"Flying to coordinate: ({tx:.3f}, {ty:.3f}, {tz:.3f})")

    start_time = time.time()
    position_reached = False
    stable_count = 0
    control_hz = 100  # Standard 100Hz control frequency
    control_interval = 1.0 / control_hz
    next_control_time = time.time()

    while time.time() - start_time < timeout:
        current_time = time.time()

        # Check position data timeout
        if not pos_manager.is_position_valid():
            logger.warning(f"Position data timeout!")

        if current_time >= next_control_time:
            # Use unified position manager to get position
            pos = pos_manager.get_position()

            error_x = abs(pos['x'] - tx)
            error_y = abs(pos['y'] - ty)
            error_z = abs(pos['z'] - tz)
            total_error = math.sqrt(error_x ** 2 + error_y ** 2 + error_z ** 2)

            # Debug output
            if stable_count % 50 == 0:
                logger.debug(f"Current position: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f}), error: {total_error:.3f}m")

            # Stability detection logic
            if position_reached:
                if total_error < tolerance:
                    stable_count += 1
                    if stable_count >= stable_steps:
                        logger.info(f"Reached waypoint (error={total_error:.3f}m, stable_count={stable_count})")
                        return True
                else:
                    # Allow some instability
                    stable_count = max(0, stable_count - 1)
            else:
                if total_error < 0.15:
                    position_reached = True
                    logger.info(f"Approaching target (error={total_error:.3f}m), entering stabilization phase")

            # Send control command - continuously send same target
            cf.commander.send_position_setpoint(tx, ty, tz, tyaw)
            collector.record_command(tx, ty, tz, tyaw)
            next_control_time += control_interval

            # Handle control time drift
            if current_time > next_control_time + control_interval:
                next_control_time = current_time + control_interval
                logger.warning("Control timing drift detected - resyncing")

        # Smart sleep to maintain control frequency
        sleep_time = min(0.001, max(0, next_control_time - time.time()))
        if sleep_time > 0:
            time.sleep(sleep_time)

    logger.warning(f"Waypoint timeout {timeout}s (final error may still be acceptable)")
    return False

# ========================= Mission execution =========================
def execute_mission_fixed(scf, waypoints: List[Tuple], collector: PureContinuousCollector) -> bool:
    """Using absolute coordinate system"""
    cf = scf.cf

    # Create unified position manager
    pos_manager = PositionManager(scf)
    pos_manager.start()

    try:
        # Wait for position stability
        time.sleep(2.0)

        # Ensure position data is valid
        if not pos_manager.is_position_valid():
            logger.error("Position manager data invalid")
            return False

        start_pos = pos_manager.get_position()
        logger.info(f"Current position: ({start_pos['x']:.3f}, {start_pos['y']:.3f}, {start_pos['z']:.3f}, {start_pos['yaw']:.1f}°)")
        logger.info("Using absolute coordinate system, ignoring takeoff position offset")

        # Unlock drone
        cf.platform.send_arming_request(True)
        time.sleep(1.0)

        # Pre-flight stability - send first waypoint position for stabilization
        first_waypoint = waypoints[0]
        target_x = first_waypoint[0]
        target_y = first_waypoint[1]
        target_z = first_waypoint[2]
        target_yaw = first_waypoint[3] if len(first_waypoint) == 4 else 90.0

        logger.info(f"Pre-flight stabilization to first waypoint: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")

        control_hz = 100
        control_interval = 1.0 / control_hz
        next_control_time = time.time()

        # Stability phase 1 - send target position
        for i in range(50):
            current_time = time.time()
            if current_time >= next_control_time:
                cf.commander.send_position_setpoint(target_x, target_y, target_z, target_yaw)
                collector.record_command(target_x, target_y, target_z, target_yaw)
                next_control_time += control_interval
            sleep_time = min(0.001, max(0, next_control_time - time.time()))
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Stability phase 2
        next_control_time = time.time()
        for i in range(100):
            current_time = time.time()
            if current_time >= next_control_time:
                cf.commander.send_position_setpoint(target_x, target_y, target_z, target_yaw)
                collector.record_command(target_x, target_y, target_z, target_yaw)
                next_control_time += control_interval
            sleep_time = min(0.001, max(0, next_control_time - time.time()))
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Execute all waypoints
        for i, waypoint in enumerate(waypoints):
            # Use absolute coordinates from waypoint directly
            target_x = waypoint[0]
            target_y = waypoint[1]
            target_z = waypoint[2]
            target_yaw = waypoint[3] if len(waypoint) == 4 else 90.0

            logger.info(f"Waypoint{i+1}: absolute coordinates({target_x:.3f}, {target_y:.3f}, {target_z:.3f}, yaw={target_yaw:.1f}°)")

            success = fly_to_point_fixed(scf, target_x, target_y, target_z, target_yaw, collector, pos_manager)
            if not success:
                logger.warning(f"Waypoint{i+1} not fully reached, continuing execution")

            # Waypoint hold
            next_control_time = time.time()
            for j in range(100):
                current_time = time.time()
                if current_time >= next_control_time:
                    cf.commander.send_position_setpoint(target_x, target_y, target_z, target_yaw)
                    collector.record_command(target_x, target_y, target_z, target_yaw)
                    next_control_time += control_interval
                sleep_time = min(0.001, max(0, next_control_time - time.time()))
                if sleep_time > 0:
                    time.sleep(sleep_time)
            time.sleep(0.5)

        # Return to origin and land
        logger.info("Returning to origin and landing...")
        fly_to_point_fixed(scf, 0, 0, 0.2, 90.0, collector, pos_manager)

        logger.info("Landing...")
        cf.commander.send_stop_setpoint()
        cf.commander.send_notify_setpoint_stop()
        logger.info("Landing complete")

        return True

    except Exception as e:
        logger.error(f"Flight error: {e}")
        return False
    finally:
        # Ensure resource cleanup
        pos_manager.stop()
        cf.commander.send_stop_setpoint()
        cf.commander.send_notify_setpoint_stop()
        cf.platform.send_arming_request(False)

# ========================= Data saving =========================
def save_training_data(data: Dict[str, Any], mission: str, waypoints: List[Tuple]):
    """Save training data"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    states = data['states']
    commands = data['commands']

    if not states:
        logger.warning("No state data")
        return

    # Calculate statistics
    positions = [[s.x, s.y, s.z] for s in states]
    velocities = [math.sqrt(s.vx**2 + s.vy**2 + s.vz**2) for s in states]

    output = {
        'mission': {
            'name': mission,
            'timestamp': timestamp,
            'waypoints': waypoints,
            'duration': states[-1].timestamp,
            'samples': len(states)
        },

        'data': {
            'states': [
                {
                    'time': s.timestamp,
                    'vector': s.to_list(),
                    'components': {
                        'pos': [s.x, s.y, s.z],
                        'vel': [s.vx, s.vy, s.vz],
                        'att': [s.roll, s.pitch, s.yaw],
                        'acc': [s.ax, s.ay, s.az]
                    }
                }
                for s in states
            ],

            'commands': commands
        },

        'stats': {
            'collection': data['stats'],
            'flight': {
                'max_vel': max(velocities) if velocities else 0,
                'mean_vel': np.mean(velocities) if velocities else 0,
                'envelope': {
                    'x': [min(p[0] for p in positions), max(p[0] for p in positions)],
                    'y': [min(p[1] for p in positions), max(p[1] for p in positions)],
                    'z': [min(p[2] for p in positions), max(p[2] for p in positions)]
                }
            }
        }
    }

    filename = f'training_data/{mission}_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Data saved: {filename}")
    logger.info(f"States: {len(states)}, Commands: {len(commands)}, Duration: {states[-1].timestamp:.1f}s")

# ========================= Flight mission patterns =========================
PATTERNS = {
    # Vertical trajectories - including ascent and descent
    'vertical_med': [(0,0,0.4), (0,0,0.8), (0,0,0.4)],  #80  160up   160down  80landing 320hovering
    'vertical_large': [(0,0,0.5), (0,0,1.0), (0,0,0.5)],  #80  160up   160down  80landing  320hovering
    'vertical_small': [(0,0,0.3), (0,0,0.6), (0,0,0.9), (0,0,0.6), (0,0,0.3)], #60  180up   120down  60landing  320hovering

    # Square trajectories - different sizes
    'square_small': [(0, 0, 0.3), (0.3, 0, 0.3), (0.3, 0.3, 0.3), (0, 0.3, 0.3), (0, 0, 0.3)],  #50      50up 50x4  50landing 5x50hovering 50down
    'square_med': [(0, 0, 0.4), (0.5, 0, 0.4), (0.5, 0.5, 0.4), (0, 0.5, 0.4), (0, 0, 0.4)],    #50      50up 50x4  50landing 5x50hovering 50down
    'square_large': [(0, 0, 0.5), (0.7, 0, 0.5), (0.7, 0.7, 0.5), (0, 0.7, 0.5), (0, 0, 0.5)],  #50      50up 50x4  50landing 5x50hovering 50down

    # Triangle trajectories - different sizes
    'triangle_small': [(0, 0, 0.6), (0.4, 0, 0.6), (0.2, 0.4, 0.6), (0, 0, 0.6)],   #50     50up 50x3   50landing 50down 4x50hovering
    'triangle_med': [(0, 0, 0.6), (0.5, 0, 0.6), (0.25, 0.5, 0.6), (0, 0, 0.6)],    #50     50up 50x3   50landing 50down 4x50hovering
    'triangle_large': [(0, 0, 0.6), (0.8, 0, 0.6), (0.4, 0.6, 0.6), (0, 0, 0.6)]    #50     50up 50x3   50landing 50down 4x50hovering
}

# ========================= Main program =========================
def main():
    parser = argparse.ArgumentParser(description='Fixed continuous data collector - absolute coordinate system')
    parser.add_argument('--uri', default=None, help='Crazyflie URI')
    parser.add_argument('--pattern', default='square', choices=PATTERNS.keys(), help='Flight pattern')
    parser.add_argument('--custom', help='Custom waypoints (format: x1,y1,z1;x2,y2,z2;...)')
    parser.add_argument('--test-only', action='store_true', help='Test connection only')
    parser.add_argument('--disable-battery', action='store_true', help='Skip battery check')

    args = parser.parse_args()

    # Parse waypoints
    if args.custom:
        waypoints = [tuple(float(p) for p in wp.split(',')) for wp in args.custom.split(';')]
        mission = 'custom'
    else:
        waypoints = PATTERNS[args.pattern]
        mission = args.pattern

    logger.info(f"=== Fixed continuous data collector ===")
    logger.info(f"Mission: {mission}")
    logger.info(f"Waypoints (absolute coordinates): {waypoints}")

    cflib.crtp.init_drivers()
    uri = args.uri or uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E6')

    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            logger.info(f"Connected: {uri}")

            # Battery check
            if not args.disable_battery:
                voltage, safe = check_battery_simple(scf)
                if not safe:
                    logger.error(f"Battery voltage too low: {voltage:.2f}V < {BATTERY_MIN_VOLTAGE}V")
                    return
                logger.info(f"Battery check passed: {voltage:.2f}V")

            # Initialize - set to origin
            logger.info("Initializing position estimator to origin...")
            scf.cf.param.set_value('kalman.initialX', 0.0)
            scf.cf.param.set_value('kalman.initialY', 0.0)
            scf.cf.param.set_value('kalman.initialZ', 0.0)
            scf.cf.param.set_value('kalman.initialYaw', math.radians(90))  # 90 degrees to radians

            reset_estimator(scf)
            wait_for_estimator_improved(scf)

            # Optimize controller parameters
            optimize_controller_for_stability(scf)
            time.sleep(1.0)  # Wait for parameters to take effect

            if args.test_only:
                logger.info("Connection test complete")

                # Show current position
                pos_manager = PositionManager(scf)
                pos_manager.start()
                time.sleep(1.0)
                pos = pos_manager.get_position()
                logger.info(f"Test position reading: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f}, {pos['yaw']:.1f}°)")
                pos_manager.stop()
            else:
                print(f"\nPreparing to execute mission: {mission}")
                print(f"Waypoint sequence: {waypoints}")
                print("Note: Using absolute coordinate system, drone will fly directly from current position to target points")
                input("Press Enter to start flight...")

                collector = PureContinuousCollector()

                try:
                    # Start data collector
                    collector.start(scf)
                    logger.info("Data collector started")

                    # Execute mission
                    success = execute_mission_fixed(scf, waypoints, collector)
                    logger.info("Flight mission complete" if success else "Flight mission partially complete")

                finally:
                    # Ensure data collector stops
                    logger.info("Stopping data collector...")
                    collector.stop()
                    data = collector.get_data()

                    if data['states']:
                        save_training_data(data, mission, waypoints)
                        logger.info(f"Successfully saved {len(data['states'])} state data points")
                        logger.info(f"Successfully saved {len(data['commands'])} control commands")
                    else:
                        logger.warning("No data collected - check data collector configuration")

    except KeyboardInterrupt:
        logger.info("User interrupted")
    except Exception as e:
        logger.error(f"Program error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()