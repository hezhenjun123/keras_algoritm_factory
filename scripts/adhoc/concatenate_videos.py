
from utilities.file_system_manipulation import directory_to_file_list,is_s3_path,copy_file
import cv2
import logging
import fire

def main(video_directory,output_path):
    file_list = sorted(directory_to_file_list(video_directory))
    total_count = len(file_list)
    output_video = None
    for file_count, file_path in enumerate(file_list, start=1):
        print(f"process video num: {file_count}/{total_count}")
        print(f"file name: {file_path}")
        if is_s3_path(file_path):
            copy_file(file_path,'tmp.avi')
            file_path = 'tmp.avi'
        input_video = cv2.VideoCapture(file_path)
        frame = input_video.read()[1]
        while frame is not None:
            if output_video is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_shape = (frame.shape[1], frame.shape[0])
                fps = input_video.get(cv2.CAP_PROP_FPS)
                output_video = cv2.VideoWriter(output_path, fourcc,
                                            fps, video_shape, True)
            output_video.write(frame)
            frame = input_video.read()[1]
        input_video.release()
    output_video.release()
    
if __name__ == '__main__':
    fire.Fire(main)