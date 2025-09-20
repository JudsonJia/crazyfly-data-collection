"""
Crazyflie LogConfig delay testing tool

Purpose: Test time delay between two LogConfig groups to provide reference for data collection optimization
Usage: python crazyflie_delay_tester.py --uri radio://0/80/2M/E7E7E7E7E6
"""

import time
import logging
import argparse
import statistics
from typing import Dict, List
import numpy as np
from collections import deque

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('delay_tester')

class DelayMeasurementCollector:
    """LogConfig delay measurement collector"""

    def __init__(self):
        self.timestamps = {
            'position_velocity': deque(maxlen=500),
            'attitude_imu': deque(maxlen=500)
        }
        self.delay_measurements = []
        self.start_time = time.time()

    def record_callback(self, timestamp, data, group_name):
        """Record timestamp of LogConfig callback"""
        current_system_time = time.time() * 1000000  # microseconds

        record = {
            'system_time_us': current_system_time,
            'cf_timestamp_us': timestamp,
            'relative_time_s': time.time() - self.start_time,
            'sample_data': data
        }

        self.timestamps[group_name].append(record)

        # Calculate delay (when both groups have data)
        if len(self.timestamps['position_velocity']) > 0 and len(self.timestamps['attitude_imu']) > 0:
            self.analyze_latest_delay()

    def analyze_latest_delay(self):
        """Analyze latest delay data"""
        pos_records = list(self.timestamps['position_velocity'])
        imu_records = list(self.timestamps['attitude_imu'])

        if len(pos_records) == 0 or len(imu_records) == 0:
            return

        # Take the latest records
        latest_pos = pos_records[-1]
        latest_imu = imu_records[-1]

        # Calculate system time delay
        system_delay_ms = abs(latest_pos['system_time_us'] - latest_imu['system_time_us']) / 1000

        # Calculate Crazyflie timestamp delay
        cf_delay_ms = abs(latest_pos['cf_timestamp_us'] - latest_imu['cf_timestamp_us']) / 1000

        measurement = {
            'system_delay_ms': system_delay_ms,
            'cf_delay_ms': cf_delay_ms,
            'pos_system_time': latest_pos['system_time_us'],
            'imu_system_time': latest_imu['system_time_us'],
            'pos_cf_time': latest_pos['cf_timestamp_us'],
            'imu_cf_time': latest_imu['cf_timestamp_us'],
            'measurement_time': time.time() - self.start_time
        }

        self.delay_measurements.append(measurement)

        # Print stats every 100 measurements
        if len(self.delay_measurements) % 100 == 0:
            self.print_delay_stats()

    def find_matching_pairs(self, time_tolerance_ms=50):
        """Find time-matched data pairs"""
        pos_records = list(self.timestamps['position_velocity'])
        imu_records = list(self.timestamps['attitude_imu'])

        matched_pairs = []
        tolerance_us = time_tolerance_ms * 1000

        for pos_record in pos_records:
            best_match = None
            min_diff = float('inf')

            for imu_record in imu_records:
                # Use Crazyflie timestamp for matching
                time_diff = abs(pos_record['cf_timestamp_us'] - imu_record['cf_timestamp_us'])
                if time_diff < min_diff and time_diff <= tolerance_us:
                    min_diff = time_diff
                    best_match = imu_record

            if best_match:
                matched_pairs.append({
                    'pos_record': pos_record,
                    'imu_record': best_match,
                    'cf_time_diff_ms': min_diff / 1000,
                    'system_time_diff_ms': abs(pos_record['system_time_us'] - best_match['system_time_us']) / 1000
                })

        return matched_pairs

    def print_delay_stats(self):
        """Print delay statistics"""
        if not self.delay_measurements:
            return

        recent = self.delay_measurements[-100:]  # Latest 100 measurements

        system_delays = [m['system_delay_ms'] for m in recent]
        cf_delays = [m['cf_delay_ms'] for m in recent]

        print(f"\n=== Delay Statistics (Latest {len(recent)} samples) ===")
        print(f"System time delay:")
        print(f"  Mean: {statistics.mean(system_delays):.2f}ms")
        print(f"  Median: {statistics.median(system_delays):.2f}ms")
        print(f"  Max: {max(system_delays):.2f}ms")
        print(f"  Min: {min(system_delays):.2f}ms")
        print(f"  Std Dev: {statistics.stdev(system_delays):.2f}ms")

        print(f"Crazyflie timestamp delay:")
        print(f"  Mean: {statistics.mean(cf_delays):.2f}ms")
        print(f"  Median: {statistics.median(cf_delays):.2f}ms")
        print(f"  Max: {max(cf_delays):.2f}ms")
        print(f"  Min: {min(cf_delays):.2f}ms")
        print(f"  Std Dev: {statistics.stdev(cf_delays):.2f}ms")

    def get_final_analysis(self):
        """Generate final analysis report"""
        if not self.delay_measurements:
            return "No delay data collected"

        # Analyze matching pairs
        matched_pairs = self.find_matching_pairs(time_tolerance_ms=100)

        system_delays = [m['system_delay_ms'] for m in self.delay_measurements]
        cf_delays = [m['cf_delay_ms'] for m in self.delay_measurements]

        pair_cf_delays = [p['cf_time_diff_ms'] for p in matched_pairs]
        pair_system_delays = [p['system_time_diff_ms'] for p in matched_pairs]

        analysis = {
            'total_samples': len(self.delay_measurements),
            'matched_pairs': len(matched_pairs),
            'matching_rate': len(matched_pairs) / max(len(self.timestamps['position_velocity']), 1) * 100,
            'system_delay_stats': {
                'mean': statistics.mean(system_delays),
                'median': statistics.median(system_delays),
                'max': max(system_delays),
                'min': min(system_delays),
                'std': statistics.stdev(system_delays) if len(system_delays) > 1 else 0
            },
            'cf_delay_stats': {
                'mean': statistics.mean(cf_delays),
                'median': statistics.median(cf_delays),
                'max': max(cf_delays),
                'min': min(cf_delays),
                'std': statistics.stdev(cf_delays) if len(cf_delays) > 1 else 0
            },
            'matched_pair_stats': {
                'cf_delays': {
                    'mean': statistics.mean(pair_cf_delays) if pair_cf_delays else 0,
                    'median': statistics.median(pair_cf_delays) if pair_cf_delays else 0,
                    'max': max(pair_cf_delays) if pair_cf_delays else 0,
                    'p95': np.percentile(pair_cf_delays, 95) if pair_cf_delays else 0
                },
                'system_delays': {
                    'mean': statistics.mean(pair_system_delays) if pair_system_delays else 0,
                    'median': statistics.median(pair_system_delays) if pair_system_delays else 0,
                    'max': max(pair_system_delays) if pair_system_delays else 0,
                    'p95': np.percentile(pair_system_delays, 95) if pair_system_delays else 0
                }
            }
        }

        return analysis

def run_delay_test(scf, test_duration_seconds=30):
    """Run delay test"""

    print(f"Starting delay test, duration {test_duration_seconds} seconds...")

    delay_collector = DelayMeasurementCollector()

    # Create position-velocity group LogConfig
    pos_vel_config = LogConfig(name='PosVelTest', period_in_ms=20)
    pos_vel_config.add_variable('kalman.stateX', 'float')
    pos_vel_config.add_variable('kalman.stateY', 'float')
    pos_vel_config.add_variable('kalman.stateZ', 'float')
    pos_vel_config.add_variable('kalman.statePX', 'float')
    pos_vel_config.add_variable('kalman.statePY', 'float')
    pos_vel_config.add_variable('kalman.statePZ', 'float')

    # Create attitude-IMU group LogConfig
    att_imu_config = LogConfig(name='AttIMUTest', period_in_ms=20)
    att_imu_config.add_variable('stabilizer.roll', 'float')
    att_imu_config.add_variable('stabilizer.pitch', 'float')
    att_imu_config.add_variable('stabilizer.yaw', 'float')
    att_imu_config.add_variable('acc.x', 'float')
    att_imu_config.add_variable('acc.y', 'float')
    att_imu_config.add_variable('acc.z', 'float')

    # Set callbacks
    pos_vel_config.data_received_cb.add_callback(
        lambda ts, data, conf: delay_collector.record_callback(ts, data, 'position_velocity')
    )
    att_imu_config.data_received_cb.add_callback(
        lambda ts, data, conf: delay_collector.record_callback(ts, data, 'attitude_imu')
    )

    try:
        # Start LogConfigs
        scf.cf.log.add_config(pos_vel_config)
        scf.cf.log.add_config(att_imu_config)
        pos_vel_config.start()
        att_imu_config.start()

        print("LogConfigs started, beginning data collection...")
        print("(Statistics will be displayed every 100 samples)")

        # Wait for test completion
        time.sleep(test_duration_seconds)

    finally:
        # Stop LogConfigs
        try:
            pos_vel_config.stop()
            att_imu_config.stop()
        except Exception as e:
            logger.warning(f"Error stopping LogConfig: {e}")

    return delay_collector

def print_recommendations(analysis):
    """Give recommendations based on test results"""
    print("\n" + "="*60)
    print("Test Result Analysis and Recommendations")
    print("="*60)

    print(f"Total samples: {analysis['total_samples']}")
    print(f"Matched data pairs: {analysis['matched_pairs']}")
    print(f"Matching rate: {analysis['matching_rate']:.1f}%")

    cf_mean_delay = analysis['matched_pair_stats']['cf_delays']['mean']
    cf_p95_delay = analysis['matched_pair_stats']['cf_delays']['p95']
    cf_max_delay = analysis['matched_pair_stats']['cf_delays']['max']

    print(f"\nCrazyflie timestamp delay:")
    print(f"  Mean delay: {cf_mean_delay:.1f}ms")
    print(f"  95th percentile delay: {cf_p95_delay:.1f}ms")
    print(f"  Max delay: {cf_max_delay:.1f}ms")

    # Give time slot recommendations
    print(f"\nRecommended time slot configuration:")

    if cf_p95_delay < 10:
        recommended_slot = 20
        print(f"  Time slot size: {recommended_slot}ms (very small delay)")
    elif cf_p95_delay < 30:
        recommended_slot = int(cf_p95_delay * 2)
        print(f"  Time slot size: {recommended_slot}ms (2x 95th percentile delay)")
    else:
        recommended_slot = int(cf_max_delay * 1.2)
        print(f"  Time slot size: {recommended_slot}ms (1.2x max delay)")

    if cf_max_delay > recommended_slot:
        print(f"  Tolerance range: ±{int(cf_max_delay - recommended_slot + 5)}ms (handle abnormal delays)")
    else:
        print(f"  Tolerance range: ±5ms (normal conditions)")

    # Data quality assessment
    print(f"\nData quality assessment:")
    if analysis['matching_rate'] > 95:
        print("  ✓ Good data synchronization quality")
    elif analysis['matching_rate'] > 80:
        print("  ⚠ Fair data synchronization quality, recommend increasing tolerance range")
    else:
        print("  ✗ Poor data synchronization quality, need to optimize LogConfig configuration")

    if cf_mean_delay < 15:
        print("  ✓ Acceptable mean delay")
    else:
        print("  ⚠ Large mean delay, may affect timing accuracy")

def main():
    parser = argparse.ArgumentParser(description='Crazyflie LogConfig delay testing tool')
    parser.add_argument('--uri', type=str, default=None, help='Crazyflie URI')
    parser.add_argument('--duration', type=int, default=30, help='Test duration (seconds)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("Crazyflie LogConfig Delay Testing Tool")
    print("-" * 50)

    # Initialize drivers
    cflib.crtp.init_drivers()
    uri = args.uri if args.uri else uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E6')

    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            print(f"Connected to Crazyflie: {uri}")

            # Run delay test
            delay_collector = run_delay_test(scf, args.duration)

            # Generate analysis report
            analysis = delay_collector.get_final_analysis()

            if isinstance(analysis, dict):
                print_recommendations(analysis)
            else:
                print(analysis)

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Exception occurred during testing")

if __name__ == "__main__":
    main()