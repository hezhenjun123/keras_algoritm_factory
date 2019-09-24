import h5py
import os
import cv2
import argparse
import multiprocessing as mp


def h5_to_png(file_path):
    base_path, file_name = os.path.split(file_path)
    save_path = base_path.replace("/uvc/", "/avi/")

    if os.path.exists(save_path) is not True:
        os.makedirs(save_path)
        print('mkdir:%s' % save_path)

    try:
        f = h5py.File(file_path, 'r')
        timestamps = f['timestamp'][:]
        data = f['data'][:]
        writer = None
        for timestamp, img in zip(timestamps, data):
            # cv2.imwrite(os.path.join(save_path, str(timestamp)+'.png'), img)
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_shape = (img.shape[1], img.shape[0])
                writer = cv2.VideoWriter(os.path.join(save_path, f"{file_name}.avi"), fourcc, 2,
                                         video_shape, True)
            writer.write(img)
        writer.release()
        f.close()
        print('Convert %s Done.' % file_path)
    except Exception as e:
        print(e)
    finally:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--video_file_path')
    parser.add_argument('-d', '--directory', help='Specific Directory to Extract')
    args = parser.parse_args()

    if args.video_file_path:
        file_path = os.path.abspath(args.video_file_path)
        print('single file:%s' % (file_path))
        h5_to_png(file_path)
    elif args.directory:
        dir_path = os.path.abspath(args.directory)
        print('folder :%s' % (dir_path))

        pool = mp.Pool(processes=16)

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if os.path.splitext(file)[-1] in ['.h5']:
                    file_path = os.path.join(root, file)
                    print('Start to Convert %s' % file_path)
                    # h5_to_png(file_path)
                    pool.apply_async(h5_to_png, (file_path,))

        pool.close()
        pool.join()
