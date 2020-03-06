import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


vcu_file_path = "/Users/Bo/Desktop/GoogleDrive/workspace/qinghai-data/2019-09-08-vcu/20190908_125456_vcu_data.csv"

vcu_data = pd.read_csv(vcu_file_path, sep=",").fillna("")

print(vcu_data.columns)

# def plot_hist(data, num):
#     plt.figure(num)
#     plt.hist(data[data > 0], bins = np.unique(data[data > 0]))



timestamp = vcu_data["timestamp"]
reel_speed_rpm = vcu_data["reel_speed_rpm"]
roller_speed_rpm = vcu_data["roller_speed_rpm"]
fan_speed_rpm = vcu_data["fan_speed_rpm"]
elevator_speed_rpm = vcu_data["elevator_speed_rpm"]
vehicle_speed_km_h = vcu_data["vehicle_speed_km_h"]
vehicle_steer_angle_degree = vcu_data["vehicle_steer_angle_degree"]
header_height_mm = vcu_data["header_height_mm"]
reel_height_percent = vcu_data["reel_speed_percent"]
loss_ratio_percent = vcu_data["loss_ratio_percent"]
gps_latitude = vcu_data["gps_latitude"]
gps_longitude = vcu_data["gps_longitude"]


# plot_hist(reel_speed_rpm, 1)
# plot_hist(roller_speed_rpm, 2)
# plot_hist(fan_speed_rpm, 3)
# plot_hist(elevator_speed_rpm, 4)
# plot_hist(vehicle_speed_km_h, 5)
# plot_hist(vehicle_steer_angle_degree, 6)
# plot_hist(header_height_mm, 7)
# plot_hist(reel_height_percent, 8)


fig, ax = plt.subplots(3, 3, tight_layout=True, figsize=(20,10))
data = reel_speed_rpm
title = "reel_speed_rpm"
ax[0][0].hist(data[data > 0], bins=np.unique(data[data > 0]))
ax[0][0].title.set_text(title)


data = roller_speed_rpm
title = "roller_speed_rpm"
ax[0][1].hist(data[data > 0], bins=np.unique(data[data > 0]))
ax[0][1].title.set_text(title)

data = fan_speed_rpm
title = "fan_speed_rpm"
ax[0][2].hist(data[data > 0], bins=np.unique(data[data > 0]))
ax[0][2].title.set_text(title)

data = elevator_speed_rpm
title = "elevator_speed_rpm"
ax[1][0].hist(data[data > 0], bins=np.unique(data[data > 0]))
ax[1][0].title.set_text(title)

data = vehicle_speed_km_h
title = "vehicle_speed_km_h"
ax[1][1].hist(data[data > 0], bins=np.unique(data[data > 0]))
ax[1][1].title.set_text(title)

data = vehicle_steer_angle_degree
title = "vehicle_steer_angle_degree"
ax[1][2].hist(data, bins=np.unique(data[data > 0]))
ax[1][2].title.set_text(title)

data = header_height_mm
title = "header_height_mm"
ax[2][0].hist(data, bins=np.unique(data[data > 0]))
ax[2][0].title.set_text(title)

data = reel_height_percent
title = "reel_height_percent"
ax[2][1].hist(data, bins=np.unique(data[data > 0]))
ax[2][1].title.set_text(title)


plt.show()


