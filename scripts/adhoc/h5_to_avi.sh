for folder in elevator_cam front_cam_rs basket_cam_1 basket_cam_2
do
    python h5_to_avi.py -d /home/ubuntu/workspace/harvester-data/QingHai_Harvest_Data/2019-08-20/data-2019-08-20-2acreage_highland_barley/uvc/"$folder"/
    python h5_to_avi.py -d /home/ubuntu/workspace/harvester-data/QingHai_Harvest_Data/2019-08-20/data-2019-08-20-6acreage_highland_barley/uvc/"$folder"/
done


for date in 2019-08-24 2019-09-06 2019-09-07 2019-09-08 2019-09-11 2019-09-13 2019-09-14 2019-09-15
do
    for folder in elevator_cam front_cam_rs basket_cam_1 basket_cam_2
    do
        python h5_to_avi.py -d /home/ubuntu/workspace/harvester-data/QingHai_Harvest_Data/"$date"/uvc/"$folder"/
    done
done