import numpy as np
import os
import cv2
import os, subprocess
import pandas as pd

def get_sec(time_str):
    """Get Seconds from time."""
    m, s = time_str.split(':')[:2]
    return int(m) * 60 + int(s)


# root = '/homes/wr301/project_storage/datasets/Amigos'
# out_dir = '/homes/wr301/project_storage/datasets/Amigossmall_larger_faces'
# excel = pd.read_csv('/homes/wr301/project_storage/amigosnet/targets.csv', index_col=[0,1,2])
# excel = pd.read_csv('/homes/wr301/project_storage/amigosnet/targets.csv', index_col=[0,1,2])
# video_sizes = pd.read_excel(
#     '/homes/wr301/project_storage/amigosnet/Video_List.xlsx',
#      index_col=1
#      )


root = '/Users/williamrichards/Desktop/datasets/amigos.nosync/Amigos'
out_dir = '/Users/williamrichards/Desktop/datasets/Amigossmall'
excel = pd.read_csv('/Users/williamrichards/Desktop/amigosnet/targets.csv', index_col=[0,1,2])


video_sizes = pd.read_excel(
    '/Users/williamrichards/Desktop/amigosnet/Video_List.xlsx',
     index_col=1
     )
size = 100
accepted_frames = 0
rejected_frames = 0
total_frames = 0


# for exp in range(1,41):
#     exp_dir = os.path.join(root, f"Exp1_P{str(exp).zfill(2)}_face_frames_new")
#     for video in os.listdir(exp_dir):
#         vid_dir = os.path.join(exp_dir, video)

#         videono = int(video.split("_")[1])

#         if  "DS_Store" not in os.path.join(vid_dir):
#             if video ==2:
#                 file_count = 0
#                 for frame in os.listdir(vid_dir):
#                     file_count += 1
#                 print (f"exp {exp} vid {video} frames {file_count}")


# print (total_frames)

for exp in range(1,41):
    exp_dir = os.path.join(root, f"Exp1_P{str(exp).zfill(2)}_face_frames_new")
    for video in os.listdir(exp_dir):
        vid_dir = os.path.join(exp_dir, video)

        if  "DS_Store" not in os.path.join(vid_dir):
            for frame in os.listdir(vid_dir):
                if (".jpg" in frame):

                    img  = cv2.imread(os.path.join(vid_dir, frame))
                    videono = int(video.split("_")[1])
                    seconds_limit = get_sec(
                        video_sizes.loc[videono]['Video_Duration'].strftime('%H:%M')
                        )
                    framenumber = int(frame.split(".")[0])
                    seconds = framenumber / 25
                    segment = (framenumber // (20*25)) + 1
                    if (
                        (framenumber % 6 == 0) &
                        (excel.index.isin([(exp, videono, segment)]).any())
                    ):

                        originalname = os.path.join(vid_dir, frame)
                        end_name = os.path.join(out_dir, f"{exp},{videono},{segment},{framenumber}.jpg")

                        rescaled_img = np.zeros((size, size, 3))

                        for channel in range(3):
                            rescaled_img[:,:,channel] = cv2.resize(
                                img[:,:,channel],
                                dsize=(size, size),
                                interpolation=cv2.INTER_AREA
                            )
                        cv2.imwrite(end_name, rescaled_img)
                        accepted_frames += 1
                        print(f'accepted_frames: {frame}')
                    else :
                        rejected_frames += 1
                        print (f'rejecting frame: {frame}')

