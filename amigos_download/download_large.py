import os
import subprocess
import logging
import pickle
import json


from detect import extract_folder

logging.basicConfig(level=logging.INFO)

with open('links.pkl', "rb") as f:
    links = pickle.load(f)


class downloader_and_unzipper:

    def __init__(downloader):
        with open('login.json', 'r') as JSON:
            json_dict = json.load(JSON)
        downloader.username = json_dict['username']
        downloader.password = json_dict['passwords']
        downloader.links = [
            f"""wget {link} --user {downloader.username} --password
            {downloader.password}"""
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

                print(f"making dir {frame_destination}")
                os.mkdir(
                    os.path.join(frame_output_file, mov_video.split(".")[0])
                    )
                cmd = (
                    f"""ffmpeg -i {start_vid_name} -r 5 {frame_destination}
                    %08d.png"""
                    )
                logging.info(cmd)
                subprocess.call(cmd, shell=True)
                os.remove(start_vid_name)

            extract_folder(frame_output_file)
            subprocess.call(f"rm -rf {unzipped_file}", shell=True)


if __name__ == "__main__":
    downloader_and_unzipper().download_and_process()
