from norfair import Detection, Tracker
import numpy as np
import cv2
from kafka import KafkaProducer
import json
from kafka.errors import KafkaError
import os
from confluent_kafka import Producer
import socket

conf = {'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
        'security.protocol': 'SASL_SSL',
        'sasl.mechanism': 'PLAIN',
        'sasl.username': os.getenv('KAFKA_API_KEY'),
        'sasl.password': os.getenv('KAFKA_API_SECRET'),
        'client.id': socket.gethostname()}

producer = Producer(conf)

# producer = KafkaProducer(
#     bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
#     security_protocol='SASL_SSL',
#     sasl_mechanism='PLAIN',
#     sasl_plain_username=os.getenv('KAFKA_API_KEY'),
#     sasl_plain_password=os.getenv('KAFKA_API_SECRET'),
#     value_serializer=lambda x: json.dumps(x).encode('utf-8')
# )

# print('Bootstrap servers:', producer.config['bootstrap_servers'])

def publish_to_kafka(topic, latitude, longitude, class_id):
    data = {'latitude': latitude, 'longitude': longitude, 'class_id': class_id}
    try:
        producer.send(topic, data).get(timeout=10)
    except KafkaError as e:
        print(f"Failed to send data to Kafka: {e}")
    finally:
        producer.flush()

tracker = Tracker(distance_function='mean_euclidean', distance_threshold=20)

data = np.load('calibration_data.npz')
K = data['K']
dist = data['DistCoeffs']
Hsat2cctv_inv = data['Hsat2cctv_inv']
T_gps2sat_inv = data['Hgps2sat_inv']

def pixel_to_gps(pixel_coordinate, K, dist, Hsat2cctv_inv, T_gps2sat_inv):
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

########################################### Main Processing Loop ###########################################

from pipeless_agents_sdk.cloud import data_stream
for payload in data_stream:
    detection_data = payload.value['data']
    detections = [create_detection(d['bbox'], d['score'], d['class_id']) for d in detection_data['data']]
    tracked_objects = tracker.update(detections=detections)
    for tracked_object in tracked_objects:
        pixel_coordinate = tracked_object.estimate[0]
        class_id = tracked_object.label
        # Velocity = tracked_object.estimate_velocity * scale # pixels/second * meters/pixel
        lat,long = pixel_to_gps(pixel_coordinate, K, dist, Hsat2cctv_inv, T_gps2sat_inv)
        publish_to_kafka('gps_coordinates', lat, long, class_id)
