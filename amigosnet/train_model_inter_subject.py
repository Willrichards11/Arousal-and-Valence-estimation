import pandas as pd
import subprocess
from AmigosDataset import *
from torchvision import datasets, transforms
import os
from torch import optim
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from existing_model.models.vgg_face import *
from sklearn.model_selection import LeaveOneGroupOut
import time
import json
import copy, random
import argparse

parser = argparse.ArgumentParser("Transform")
parser.add_argument('--adapted', dest='adapted', action='store_true')
parser.set_defaults(adapted=False)
parser.add_argument('--reduced', dest='reduced', action='store_true')
parser.set_defaults(reduced=False)
parser.add_argument('--hog', dest='hog', action='store_true')
parser.set_defaults(hog=False)
parser.add_argument('--affectnet', dest='affectnet', action='store_true')
parser.set_defaults(affectnet=False)
args = parser.parse_args()

print (f"adapted: {args.adapted}")
print (f"reduced: {args.reduced}")
print (f"hog: {args.hog}")
print (f"affectnet: {args.affectnet}")

time.sleep(3)

torch.cuda.set_device(1)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
criterion = nn.MSELoss()

means = (0.0, 0.0, 0.0)
stds = (255.0, 255.0, 255.0)
weight_decay = 5e-4
momentum = 0.9
lr = 0.001

train_trans = transforms.Compose([
            RandomMirror(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
            ])

val_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
            ])

file_path = "/homes/wr301/project_storage/affectnet/models/fernet"\
            "_2020_08_18-142245/fernet_2020_08_18-142245-E15-b64-transfer.pth"


size = {
    True: 1296,
    False: 656
}

participants = set(
    [
        int(file.split(',')[0]) for file in
        os.listdir("/homes/wr301/project_storage/datasets/Amigossmall/Images")]
    )
participants = list(participants)

for participant in participants:

    if args.adapted == True:
        save_dir = f"inter,adapted-{args.adapted},reducedsecond-{args.reduced},hog-{args.hog},affect-{args.affectnet},{lr},{momentum},{weight_decay}"
    else:
        save_dir = f"inter,adapted-{args.adapted},reducedsecond-{args.reduced},affect-{args.affectnet},{lr},{momentum},{weight_decay}"

    if not os.path.exists(f'{save_dir}/{participant}.json'):

        participant = int(participant)
        if args.reduced == True:
            participant_videos = {
                idx : file for idx, file in enumerate(
                    os.listdir("/homes/wr301/project_storage/datasets/Amigossmall/ImagesOversamplesecondattempt")
                    )
                if int(file.split(",")[0]) == participant
            }
        else:
            participant_videos = {
                idx : file for idx, file in enumerate(
                    os.listdir("/homes/wr301/project_storage/datasets/Amigossmall/Images")
                    )
                if int(file.split(",")[0]) == participant
            }

        mapping_df = pd.DataFrame(
            participant_videos, index=["file"]
            ).T
        mapping_df['video'] = mapping_df["file"].apply(
            lambda x : int(x.split(",")[1])
            )
        if args.adapted:
            data = AmigosDatasetInterSubjectAdapted(
                "targets.csv",
                 "/homes/wr301/project_storage/datasets/Amigossmall",
                participant,
                args.hog,
                args.reduced
             )
        else:
            data = AmigosDatasetInterSubject(
                "targets.csv",
                 "/homes/wr301/project_storage/datasets/Amigossmall",
                participant,
                args.reduced
             )
        for split, video in enumerate(set(mapping_df.video.values)):
            video = int(video)
            print (f"excluding video: {video}")
            print(split, video)
            train_idxs = mapping_df[mapping_df.video != video].index.values
            test_idxs = mapping_df[mapping_df.video == video].index.values

            split_start = time.time()

            if args.adapted == True:
                print ('initialise_model_affect_adapted')
                additional_size = size[args.hog]
                net = initialise_model_affect_adapted(file_path, additional_size, args.affectnet)
                trainSet = SubsetAdapted(data, train_idxs, transform=train_trans)
                testSet = SubsetAdapted(data, test_idxs, transform=val_trans)

            elif not (args.adapted == True):
                print ('initialise_model_affect')
                net = initialise_model_affect(file_path, args.affectnet)
                trainSet = Subset(data, train_idxs, transform=train_trans)
                testSet = Subset(data, test_idxs, transform=val_trans)

            optimizer = optim.SGD(
                net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
                )

            trainloader = DataLoader(
                trainSet, batch_size=40, shuffle=True, sampler=None,
                batch_sampler=None,
                )

            testloader = DataLoader(
                testSet, batch_size=1, shuffle=False, sampler=None,
                batch_sampler=None,
                )

            epoch = 0
            threshold = 2
            cont = True
            best_val_score = eval_model(net, testloader, criterion)
            print(best_val_score)
            running_count = 0

            while cont == True:
                epoch += 1
                print(epoch)

                net = run_epoch(net, trainloader, optimizer, criterion)
                eval_score = eval_model(net, testloader, criterion)
                print (eval_score)
                if eval_score > best_val_score :
                    if running_count == threshold:
                        cont = False
                        print (f'ended on epoch {epoch}')
                    else:
                        running_count += 1
                else:
                    best_val_score = eval_score
                    best_model = copy.deepcopy(net)
                    # running_count = 0

            best_model.eval()
            epoch_loss = np.array([], dtype=np.float32)
            targets = []
            predicted = []

            if not (args.adapted == True):
                for i, testdata in enumerate(testloader, 1):
                    inputs, labels = testdata[0].cuda().float(), testdata[1].cuda().float()
                    outputs = best_model(inputs)
                    targets.append(labels.detach().cpu().numpy())
                    predicted.append(outputs.detach().cpu().numpy())
            else:
                for i, testdata in enumerate(testloader, 1):
                    images  = testdata[0].cuda().float()
                    additional  = testdata[1].cuda().float()
                    labels = testdata[2].cuda().float()

                    outputs = best_model(images, additional)
                    targets.append(labels.detach().cpu().numpy())
                    predicted.append(outputs.detach().cpu().numpy())

            predicted_arousal = np.array([x[0][0] for x in predicted])
            predicted_valence = np.array([x[0][1] for x in predicted])

            actual_arousal = np.array([x[0][0] for x in targets])
            actual_valence = np.array([x[0][1] for x in targets])

            err_arousal = predicted_arousal - actual_arousal
            err_valence = predicted_valence - actual_valence

            data_to_dump = {
                "_id": video,
                "leftout_video": video,
                "predicted_arousal": predicted_arousal.tolist(),
                "actual_arousal": actual_arousal.tolist(),
                "predicted_valence": predicted_valence.tolist(),
                "actual_valence": actual_valence.tolist(),
                "cor_arousal": np.corrcoef(predicted_arousal, actual_arousal).tolist(),
                "cor_valence": np.corrcoef(predicted_valence, actual_valence).tolist(),
                "err_arousal": err_arousal.tolist(),
                "err_valence": err_valence.tolist(),
                "split_time_hrs": (time.time() - split_start) / (60 * 60)
                }


            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            if args.adapted == True:
                with open(f'{save_dir}/{participant},{video}.json', 'w') as fp:
                    json.dump(data_to_dump, fp)

            if not args.adapted == True:
                with open(f'{save_dir}/{participant},{video}.json', 'w') as fp:
                    json.dump(data_to_dump, fp)
    else:
        print (f"Skipping participant {participant}")
        time.sleep(0.5)
        continue

