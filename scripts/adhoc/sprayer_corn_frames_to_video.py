from utilities.frames_to_vid import frames_to_vid

def img_filter(f: str) -> bool:
    return f.startswith("_Color") and f.endswith(".png")


def img_sort_key(f: str) -> int:
    return int(f.replace("_Color_", "").replace(".png", ""))


imgs_dir = "/s3/zoomlion-dev-data/sprayer/corn-bbox-seg-v0.1/corn dataset"
output_avi_path = "out.avi"
fps = 10
width_dim = 848
height_dim = 480

frames_to_vid(imgs_dir, img_filter, img_sort_key, fps, width_dim, height_dim, output_avi_path)