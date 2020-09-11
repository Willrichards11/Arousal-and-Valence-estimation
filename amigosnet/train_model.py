from AmigosDataset import (
    eval_model, run_epoch, Subset, SubsetAdapted, AmigosDataset,
    AmigosDatasetAdapted
    )
from torchvision import transforms
import os
import torch
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
from existing_model.models.vgg_face import (
    initialise_model_affect, RandomMirror, initialise_model_affect_adapted,
    )
from sklearn.model_selection import LeaveOneGroupOut
import time
import json
import torch.nn as nn
import copy
import argparse

# command line arguments
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


print(f"adapted: {args.adapted}")
print(f"reduced: {args.reduced}")
print(f"hog: {args.hog}")
print(f"affectnet: {args.affectnet}")


time.sleep(3)

# set cuda gpu and default type
torch.cuda.set_device(1)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

group_kfold = LeaveOneGroupOut()
criterion = nn.MSELoss()

# additional feature shapes - true hog false qlzm
size = {
    True: 1296,
    False: 656
}


if args.adapted is True:
    print('using adapted dataset')
    data = AmigosDatasetAdapted(
        "targets.csv",
        "/homes/wr301/project_storage/datasets/Amigossmall",
        hog=args.hog,
        reduced_dataset=args.reduced
         )

else:
    print('using not adapted dataset')

    data = AmigosDataset(
        "targets.csv",
        "/homes/wr301/project_storage/datasets/Amigossmall",
        args.reduced
        )

groups = np.array([int(file.split(',')[0]) for file in data.map.values()])
# generate all possible participant ids
participants = list(set(groups))
print(participants)

means = (0.0, 0.0, 0.0)
stds = (255.0, 255.0, 255.0)

# define transformations used for model training
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

results = {}

weight_decay = 5e-4
momentum = 0.9
lr = 0.001


# cross validate across all participants
for participant in participants:
    if args.adapted is True:
        save_dir = f"""loso,adapted-{args.adapted},reducedsecond-
        {args.reduced},hog-{args.hog},affect-{args.affectnet},{lr},{momentum},
        {weight_decay}"""
    else:
        save_dir = f"""loso,adapted-{args.adapted},reducedsecond-{args.reduced}
        ,affect-{args.affectnet},{lr},{momentum},{weight_decay}"""

    # if we haven't already done this participant
    if not os.path.exists(f'{save_dir}/{participant}.json'):

        participant = int(participant)
        train_idxs = np.where(groups != participant)[0]
        test_indxs = np.where(groups == participant)[0]
        split_start = time.time()
        print(args.adapted)

        if args.adapted is True:
            print('initialise_model_affect_adapted')
            additional_size = size[args.hog]
            net = initialise_model_affect_adapted(
                file_path, additional_size, args.affectnet
                )
            trainSet = SubsetAdapted(data, train_idxs, transform=train_trans)
            testSet = SubsetAdapted(data, test_indxs, transform=val_trans)

        elif args.adapted is not True:
            print('initialise_model_affect')
            net = initialise_model_affect(file_path, args.affectnet)
            trainSet = Subset(data, train_idxs, transform=train_trans)
            testSet = Subset(data, test_indxs, transform=val_trans)

        optimizer = optim.SGD(
            net.parameters(), lr=lr, momentum=momentum,
            weight_decay=weight_decay
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

        while cont is True:
            epoch += 1
            print(epoch)

            net = run_epoch(net, trainloader, optimizer, criterion)
            eval_score = eval_model(net, testloader, criterion)
            print(eval_score)
            if eval_score > best_val_score:
                if running_count == threshold:
                    cont = False
                    print(f'ended on epoch {epoch}')
                else:
                    running_count += 1
            else:
                best_val_score = eval_score
                best_model = copy.deepcopy(net)

        # the below code is used to save model results to json format
        # these results are analysed using jupyter notebooks later on
        best_model.eval()
        epoch_loss = np.array([], dtype=np.float32)
        targets = []
        predicted = []
        if args.adapted is not True:
            for i, testdata in enumerate(testloader, 1):
                inputs, labels = (
                    testdata[0].cuda().float(), testdata[1].cuda().float()
                    )
                outputs = best_model(inputs)
                targets.append(labels.detach().cpu().numpy())
                predicted.append(outputs.detach().cpu().numpy())
        else:
            for i, testdata in enumerate(testloader, 1):
                images = testdata[0].cuda().float()
                additional = testdata[1].cuda().float()
                labels = testdata[2].cuda().float()

                outputs = best_model(images, additional)
                targets.append(labels.detach().cpu().numpy())
                predicted.append(outputs.detach().cpu().numpy())

        predicted_arousal = np.array([x[0][0] for x in predicted])
        predicted_valence = np.array([x[0][1] for x in predicted])

        actual_arousal = np.array([x[0][0] for x in targets])
        actual_valence = np.array([x[0][1] for x in targets])

        data_to_dump = {
            "_id": participant,
            "leftout": participant,
            "predicted_arousal": predicted_arousal.tolist(),
            "actual_arousal": actual_arousal.tolist(),
            "predicted_valence": predicted_valence.tolist(),
            "actual_valence": actual_valence.tolist(),
            "split_time_hrs": (time.time() - split_start) / (60 * 60)
            }

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        with open(f'{save_dir}/{participant}.json', 'w') as fp:
            json.dump(data_to_dump, fp)
    else:
        print(f"Skipping participant {participant}")
        time.sleep(0.5)
        continue
