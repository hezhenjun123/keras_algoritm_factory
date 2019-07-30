import pandas as pd
import numpy as np
import os
from boto3 import client


def give_split(name, train_generated_only=False):
    if (train_generated_only or
        np.random.uniform() > (.5)) and \
        'generated' not in name:
        return 'valid'
    else:
        return 'train'


conn = client('s3')
seg_label_paths = []
img_paths = []
for key_dir in [
        'wheat_dmg_segmentation',
]:
    segmaps_full_paths = conn.list_objects(
        Bucket='zoomlion-sample', Prefix=f'{key_dir}/segmaps')['Contents']
    seg_label_paths.extend([
        f'{key_dir}/segmaps/' + os.path.basename(key['Key'])
        for key in segmaps_full_paths
    ])
    img_full_paths = conn.list_objects(Bucket='zoomlion-sample',
                                       Prefix=f'{key_dir}/imgs')['Contents']
    img_paths.extend([
        f'{key_dir}/imgs/' + os.path.basename(key['Key'])
        for key in img_full_paths
    ])

# filter prism extras
img_paths = [x for x in img_paths if '_label.png' not in x and x[-1] != '/']
seg_label_paths = [
    x for x in seg_label_paths if '_label.png' not in x and x[-1] != '/'
]

assert len(img_paths) == len(
    seg_label_paths), 'Something is not matching in lengths'

assert len(
    set([os.path.basename(x) for x in img_paths])
    - set([os.path.basename(x).replace('_segmap', '')
           for x in img_paths])) == 0, 'Names do not match up'

label_name = [""] * len(img_paths)
label = [""] * len(img_paths)

splits = [give_split(name) for name in seg_label_paths]

df = pd.DataFrame(list(
    zip(img_paths, seg_label_paths, label_name, label, splits)),
                  columns=[
                      'image_path', 'segmentation_path', 'label_name',
                      'image_level_label', 'split'
                  ])
print(df.head())
print("Length of the full dataframe is", len(df))
print("Train/Validation split is\n", df['split'].value_counts())

os.makedirs('data/', exist_ok=True)

df.to_csv('data/train_segmentation_new.csv', sep='\t', index=False)
print('Dataframe saved to data/train_segmentation_new.csv')
