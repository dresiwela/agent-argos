from pipeless_agents_sdk.cloud import data_stream
from paho.mqtt import client as mqtt_client
import certifi
import json
import numpy as np
import cv2
import os
from norfair import Detection, Tracker
from geopy.distance import geodesic
from geographiclib.geodesic import Geodesic
import time

prev_coordinates = {}

def calculate_speed(coord1, coord2, dt):
    if dt <= 0:
        return 0.0
    distance_m = geodesic(coord1, coord2).meters
    return distance_m / dt

def calculate_bearing(coord1, coord2):
    result = Geodesic.WGS84.Inverse(coord1[0], coord1[1], coord2[0], coord2[1])
    return result['azi1'] % 360

def pixel_to_gps(pixel_coordinate, K, dist, Hsat2cctv_inv, T_gps2sat_inv):
    # Convert pixel coordinates to GPS coordinates
    pixel_coordinate = np.float64(pixel_coordinate).reshape(-1, 1, 2)
    pixel_coordinate = cv2.undistortPoints(pixel_coordinate, K, dist, None, K)
    sat_pixel_coordinate = cv2.perspectiveTransform(pixel_coordinate, Hsat2cctv_inv)
    gps_coordinate = cv2.transform(sat_pixel_coordinate, T_gps2sat_inv).flatten()[:2]
    return gps_coordinate[0], gps_coordinate[1]

def create_detection(bbox, score, label):
    ground_contact_x = bbox[0] + bbox[2] / 2
    ground_contact_y = bbox[1] + bbox[3]
    ground_contact_point = np.array([[ground_contact_x, ground_contact_y]])
    return Detection(points=ground_contact_point, scores=np.array([score]), label=label)

def connect_mqtt():
    # Connect to MQTT Broker
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}")

    def on_disconnect(client, userdata, rc):
        if rc != 0:
            print("Unexpected disconnection.")

    client = mqtt_client.Client(client_id=client_id)
    client.tls_set(ca_certs=certifi.where())  # Using certifi's CA bundle
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    client.connect(broker, port)
    client.loop_start()
    return client

# Load calibration data
data = np.load('calibration_data.npz')
K = data['K']
dist = data['DistCoeffs']
Hsat2cctv_inv = data['Hsat2cctv_inv']
T_gps2sat_inv = data['Hgps2sat_inv']

broker = os.getenv('mqtt_broker')
port = 8883
topic = 'argos/gps'
client_id = 'python-mqtt-argos'
username = os.getenv('username')
password = os.getenv('password')

client = connect_mqtt()

tracker = Tracker(distance_function='mean_euclidean', distance_threshold=40)

for payload in data_stream:
    detection_data = payload.value['data']
    detections = [create_detection(d['bbox'], d['score'], d['class_id']) for d in detection_data]
    tracked_objects = tracker.update(detections=detections)

    now = time.time()
    frame_data = {}

    lat_offset = 6.424739041221983e-05
    long_offset = 7.77840633361393e-05
    for tracked_object in tracked_objects:
        pixel_coordinate = tracked_object.estimate[0]
        class_id = tracked_object.label
        obj_id = tracked_object.id
        lat, long = pixel_to_gps(pixel_coordinate, K, dist, Hsat2cctv_inv, T_gps2sat_inv)
        lat = lat + lat_offset
        long = long + long_offset
        print(pixel_coordinate,lat,long)

        if obj_id not in prev_coordinates:
            prev_coordinates[obj_id] = (lat, long, now)
            continue

        prev_lat, prev_long, prev_time = prev_coordinates[obj_id]
        speed = calculate_speed((prev_lat, prev_long), (lat, long), now - prev_time)
        bearing = calculate_bearing((prev_lat, prev_long), (lat, long))
        prev_coordinates[obj_id] = (lat, long, now)

        frame_data[obj_id] = {
            'latitude': lat,
            'longitude': long,
            'class_id': class_id,
            'speed': np.round(speed, 2),
            'orientation': bearing
        }

    if client.is_connected():
        client.publish(topic, json.dumps(frame_data), qos=1)
    else:
        print("Not connected to MQTT Broker. Attempting to reconnect.")
        client.reconnect()

print("Done")
# client.loop_stop()
# client.disconnect()




# unique_id_counter = 0
# for payload in data_stream:
#     detection_data = payload.value['data']
#     detections = [create_detection(d['bbox'], d['score'], d['class_id']) for d in detection_data]
#     frame_data = []

#     for detection in detections:
#         pixel_coordinate = detection.points[0]
#         class_id = detection.label
#         lat, long = pixel_to_gps(pixel_coordinate, K, dist, Hsat2cctv_inv, T_gps2sat_inv)

#         unique_id_counter += 1
#         detection_id = unique_id_counter

#         frame_data.append({
#             'latitude': lat,
#             'longitude': long,
#             'id': detection_id,
#             'class_id': class_id,
#             'speed': 0.0,  # Speed and orientation are placeholders
#             'orientation': 0.0
#         })

#     if client.is_connected():
#         client.publish(topic, json.dumps(frame_data), qos=1)
#     else:
#         print("Not connected to MQTT Broker. Attempting to reconnect.")
#         client.reconnect()