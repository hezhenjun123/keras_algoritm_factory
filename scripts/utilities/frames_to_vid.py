import os
import cv2
from os import listdir
from typing import Callable, List


def frames_to_vid(imgs_dir: str, img_filter: Callable[[str], bool], img_sort_key: Callable[[str], int], fps: int, px_width: int, px_height: int,  output_avi_path: str):
    if os.path.exists(output_avi_path):
        os.remove(output_avi_path)

    frames: List[str] = [f for f in listdir(imgs_dir) if img_filter(f) ]
    frames.sort(key=img_sort_key)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_avi_path, fourcc, fps, (px_width, px_height))

    for f in frames:
        absf = os.path.join(imgs_dir,f)
        frame = cv2.imread(absf)
        print(frame.shape, absf)
        out.write(frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break

    out.release()
    print("Finished writing ", output_avi_path)

