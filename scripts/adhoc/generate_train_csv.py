import glob
absolute_path = "/Users/Bo/Desktop/Data/harvester/lodging/lodgingwheat/lodgingwheat/lodgingwheat/"
folder_str = "111APPLE"
output_path = f"/Users/Bo/Desktop/Data/harvester/lodging/lodgingwheat/lodgingwheat/lodgingwheat/train_raw_{folder_str}.csv"
schema = "image_path,seg_label_path,split,label_name,image_level_label"

file_list = sorted(glob.glob(absolute_path + f"{folder_str}/*.JPG"))
# print(len(file_list))
# print(file_list[2000:2100])
# print(file_list[2000].replace(absolute_path, ""))
seg_path = "seg/ab73b225-031c-4110-ac70-40d1988aa576.png"
with open(output_path, 'w') as fw:
    fw.write(schema + '\n')
    for file in file_list:
        curr_file = file.replace(absolute_path, "")
        fw.write(f"{curr_file},{seg_path},raw,,\n")