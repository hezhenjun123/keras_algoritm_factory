import h5py
import os
import cv2
import argparse
import multiprocessing as mp
import shutil

def h5_to_png(file_path):
    base_path, file_name = os.path.split(file_path)
    #h5 doesn't support reading from s3 mounted bucket
    # so we copy the file
    shutil.copyfile(file_path,file_name)
    save_path_avi = base_path.replace("/uvc/", "/avi/")
    save_path_h5 = base_path.replace("/uvc/", "/avi_frame_timestamps/")
    for save_path in [save_path_avi,save_path_h5]:
        if os.path.exists(save_path) is not True:
            os.makedirs(save_path)
            print('mkdir:%s' % save_path)

    try:
        f = h5py.File(file_name, 'r')
        data = f['data'][:]
        writer = None
        for img in data:
            # cv2.imwrite(os.path.join(save_path, str(timestamp)+'.png'), img)
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_shape = (img.shape[1], img.shape[0])
                writer = cv2.VideoWriter(os.path.join(save_path_avi, f"{file_name}.avi"), fourcc, 2,
                                         video_shape, True)
            writer.write(img)
        writer.release()
        f_new = h5py.File(f"timestamps_{file_name}",'w-')
        f.copy('timestamp',f_new)
        f_new.close()
        f.close()
        shutil.move(f"timestamps_{file_name}",os.path.join(save_path_h5, f"{file_name}"))
        print('Convert %s Done.' % file_path)
    except Exception as e:
        print(e)
    finally:
        os.remove(file_name)
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
            if 'avi_frame_timestamps' in root: continue
            for file in files:
                if os.path.splitext(file)[-1] in ['.h5']:
                    file_path = os.path.join(root, file)
                    print('Start to Convert %s' % file_path)
                    # h5_to_png(file_path)
                    pool.apply_async(h5_to_png, (file_path,))

        pool.close()
        pool.join()
