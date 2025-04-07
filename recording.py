from datetime import datetime
from meshcat.animation import convert_frames_to_video

now = datetime.now()
timestamp = now.strftime("%m%d%Y%H%M%S")

filepath = "/Users/bugart/Downloads/meshcat_1743275923052.tar"
convert_frames_to_video(filepath, output_path=f"videos/{timestamp}.mp4", overwrite=True)
