from norfair import Detection, Tracker
import numpy as np
import cv2
from kafka import KafkaProducer
import json
from kafka.errors import KafkaError

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    retries=5,
    # acks='all',  # Ensure all replicas ack the message
    # linger_ms=100,  # Batch messages to reduce requests made to the server
    # max_in_flight_requests_per_connection=5  # Maintain order within each partition
)

def publish_to_kafka(topic, data):
    try:
        producer.send(topic, data).get(timeout=10)  # Wait for send confirmation
    except KafkaError as e:
        print(f"Failed to send data to Kafka: {e}")
    finally:
        producer.flush()

tracker = Tracker(distance_function='mean_euclidean', distance_threshold=20)

data = np.load('calibration_data.npz')
K = data['K']
dist = data['dist']
Hsat2cctv_inv = data['Hsat2cctv_inv']
T_gps2sat_inv = data['Hgps2sat_inv']

def pixel_to_gps(pixel_coordinate, K, dist, Hsat2cctv_inv, T_gps2sat_inv):
    pixel_coordinate = np.float64(pixel_coordinate).reshape(-1, 1, 2)
    pixel_coordinate = cv2.undistortPoints(pixel_coordinate, K, dist, None, K)
    sat_pixel_coordinate = cv2.perspectiveTransform(pixel_coordinate, Hsat2cctv_inv)
    gps_coordinate = cv2.transform(sat_pixel_coordinate, T_gps2sat_inv).flatten()[:2]
    return gps_coordinate

def create_detection(bbox, score, label):
    ground_contact_x = bbox[0] + bbox[2] / 2
    ground_contact_y = bbox[1] + bbox[3]
    ground_contact_point = np.array([[ground_contact_x, ground_contact_y]])
    return Detection(points=ground_contact_point, scores=np.array([score]), label=label)

# Main processing loop
from pipeless_agents_sdk.cloud import data_stream
for payload in data_stream:
    detection_data = payload.value['data']
    detections = [create_detection(d['bbox'], d['score'], d['class_id']) for d in detection_data['data']]
    tracked_objects = tracker.update(detections=detections)
    for tracked_object in tracked_objects:
        pixel_coordinate = tracked_object.estimate[0]
        gps_coordinate = pixel_to_gps(pixel_coordinate, K, dist, Hsat2cctv_inv, T_gps2sat_inv)
        publish_to_kafka('gps_coordinates', {'gps': gps_coordinate.tolist()})