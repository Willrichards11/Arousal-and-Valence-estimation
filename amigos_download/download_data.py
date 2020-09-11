import os
import subprocess
import logging
import json
from detect import extract_folder

logging.basicConfig(level=logging.INFO)


class downloader_and_unzipper:
    def __init__(downloader):
        with open('login.json', 'r') as JSON:
            json_dict = json.load(JSON)

        downloader.username = json_dict['username']
        downloader.password = json_dict['passwords']
        downloader.links = downloader.gen_links()

    def gen_links(downloader):
        link_dict = {}
        for a_number in range(1, 21):

            number_str = str(a_number)
            zero_filled_number = number_str.zfill(2)

            link_dict[
                    f"""wget http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/
                    data/Exp1_P{zero_filled_number}_face.zip --user
                    {downloader.username} --password={downloader.password}"""
                ] = "Exp1_P" + zero_filled_number + "_face.zip"
        return link_dict

    def download_and_process(downloader):

        for link in downloader.links.keys():
            filename = downloader.links[link]
            unzipped_file = filename.split(".")[0]
            frame_output_file = unzipped_file + "_frames_new"

            if not os.path.isfile(filename):
                logging.info("downloading file: " + filename)
                subprocess.call(link, shell=True)
            logging.info("unzipping file: " + filename)
            os.system("7za x " + filename)
            os.remove(filename)
            logging.info("converting videos in: " + unzipped_file)

            for mov_video in os.listdir(unzipped_file):
                start_vid_name = os.path.join(unzipped_file, mov_video)
                end_vid_name = os.path.join(
                    unzipped_file, mov_video.split(".")[0] + ".mp4"
                    )

                cmd = (
                    f"""ffmpeg -r 25 -i {start_vid_name} -vcodec h264
                    -acodec mp2 {end_vid_name}"""
                    )
                logging.info(cmd)
                subprocess.call(cmd, shell=True)
                os.remove(start_vid_name)
            os.mkdir(frame_output_file)
            subprocess.call(
                f"""python AMIGOS_CODE/face_extractor/extract.py --d
                {unzipped_file} --o {frame_output_file}""",
                shell=True
                )
            print(os.listdir(frame_output_file))
            extract_folder(frame_output_file)
            subprocess.call(f"rm -rf {unzipped_file}", shell=True)


if __name__ == "__main__":
    downloader_and_unzipper().download_and_process()
