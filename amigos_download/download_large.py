#!/usr/bin/env python
# coding: utf-8


import os, subprocess, logging, zipfile
from shutil import rmtree
from detect import *

logging.basicConfig(level = logging.INFO)


links = [
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L01_Indiv_N09_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L02_Indiv_N23_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L04_Indiv_N25_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L05_Indiv_N13_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L07_Indiv_N31_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L08_Indiv_N26_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L12_Indiv_N30_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L13_Indiv_N34_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L14_Indiv_N35_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L15_Indiv_N37_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L16_Indiv_N33_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L17_Indiv_N20_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L18_Indiv_N36_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L19_Indiv_N19_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L20_Indiv_N38_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L21_Indiv_N39_face.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Exp2_L22_Indiv_N40_face.zip"
]


class downloader_and_unzipper:
    def __init__(downloader):
        with open('login.json', 'r') as JSON:
            json_dict = json.load(JSON)
        downloader.username = json_dict['username']
        downloader.password = json_dict['passwords']
        downloader.links = [
            f"wget {link} --user {downloader.username} --password {downloader.password}"
            for link in links
        ]


    def download_and_process(downloader):

        for link in downloader.links:
            filename = link.split("/")[-1].split(" ")[0]
            unzipped_file = filename.split(".")[0]
            frame_output_file = unzipped_file + "_frames"
            os.mkdir(frame_output_file)

            if not os.path.isfile(filename):
                logging.info("downloading file: " + filename)
                subprocess.call(link, shell=True)

            logging.info("unzipping file: " + filename)
            os.system("7za x " + filename)
            os.remove(filename)
            logging.info("converting videos in: " + unzipped_file)

            for mov_video in os.listdir(unzipped_file):
                start_vid_name = os.path.join(unzipped_file, mov_video)
                frame_destination = os.path.join(
                    frame_output_file,
                    mov_video.split(".")[0]
                    )

                print (f"making dir {frame_destination}")
                os.mkdir(os.path.join(frame_output_file, mov_video.split(".")[0]))
                cmd = f"ffmpeg -i {start_vid_name} -r 5 {frame_destination}%08d.png"
                logging.info(cmd)
                subprocess.call(cmd, shell=True)
                os.remove(start_vid_name)

            extract_folder(frame_output_file)
            subprocess.call(f"rm -rf {unzipped_file}", shell=True)

if __name__=="__main__":
    downloader_and_unzipper().download_and_process()
