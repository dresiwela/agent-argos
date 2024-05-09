import numpy as np
import boto3

data = DynamoDB

def parse_data(data):
    return np.array([[float(item['N']) for item in sublist['L']] for sublist in data['L']])

# Convert each data group to numpy arrays
Hsat2cctv = parse_data(data['Hsat2cctv'])
K = parse_data(data['K'])
DistCoeffs = np.array([float(item['N']) for item in data['DistCoeffs']['L']])
Hgps2sat = parse_data(data['Hgps2sat'])

Hsat2cctv_inv = np.linalg.pinv(Hsat2cctv)
Hgps2sat_inv = np.linalg.pinv(Hgps2sat)

# Save arrays to an NPZ file
np.savez('calibration_data.npz', Hsat2cctv_inv=Hsat2cctv_inv, K=K, DistCoeffs=DistCoeffs, Hgps2sat_inv=Hgps2sat_inv)

print("Data has been successfully saved to 'calibration_data.npz'.")