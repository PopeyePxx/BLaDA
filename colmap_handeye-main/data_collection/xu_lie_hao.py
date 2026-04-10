import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

for dev in devices:
    print("Found device:")
    print("  Name        :", dev.get_info(rs.camera_info.name))
    print("  Serial      :", dev.get_info(rs.camera_info.serial_number))