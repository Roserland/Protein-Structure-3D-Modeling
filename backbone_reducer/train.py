import os, sys
sys.path.append('..')

import json
import random
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score,jaccard_score,accuracy_score,confusion_matrix,roc_auc_score
from backbone_dataset import CaBackboneSet

from models.UNet3D import UNet3D
from models.loss import WeightedCrossEntropyLoss, FocalLoss
# from prediction import prediction_and_visualization
# from rmsd import rmsd_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_auc_score
from tqdm import tqdm
from collections import Counter

from utils.decoder import quant_scores, time_counter
# sys.path.append('..')
os.chdir(sys.path[0])


SEED = 20487 # 123
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(epoch):
    test_dir = '/mnt/data/Storage4/mmy/CryoEM_0112_ex_test'
    filenames = os.listdir(test_dir)
    model_path = './checkpoints/0112_0226_'+args.type+'_'+args.dataset+'_checkpoint_epoch_'+str(epoch)
    for pdb_id in filenames[0:12]:
        normalized_map = os.path.join(test_dir,pdb_id,'experiment','normalized_map.mrc')
        if os.path.exists(normalized_map):
            ca_confidence_map_path = './tmp/'+pdb_id+'_'+str(epoch)+'_ca_test.mrc'
            prediction_and_visualization(model_path,normalized_map,ca_confidence_map_path,pdb_id)
            
            pdb_file_path = os.path.join(test_dir,pdb_id,pdb_id+'_ca.pdb')
            rmsd_score(ca_confidence_map_path,pdb_file_path,pdb_id)


@time_counter()
def eval_all(valid_dataloader, model, thres=0.5):
    """
    Eval with .sigmoid() but not .sfotmax()
    """
    with torch.no_grad():
        pred_labels_all = np.array([])
        pred_scores_all = np.array([])
        gt_labels_all  = np.array([])

        res = {
            "pixel_nums": 0,
            "recall": 0,
            "precision": 0,
            "mAP": 0,
            "F1-Score": 0,
            "AUC": 0,
            "num_iters": 0
        }
        confusion_mat = np.zeros([2, 2])
        num_iters = len(valid_dataloader)

        logging.info('starting new evaluation')
        for (img, label) in tqdm(valid_dataloader):
            img = img.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.long)
            img = torch.reshape(img, (-1, 1, 64, 64, 64))
            logits = model(img)

            logits = logits[:, :, 7:57, 7:57, 7:57]
            label  = label[:, 7:57, 7:57, 7:57]

            # using log_softmax to get confidence score
            prob = torch.softmax(logits, dim=1)  # softmax on logits
            pred_scores, pred_labels = prob.max(dim=1)
            pred_scores = prob[:, 1, :, :, :]

            # using sigmoid to get confidence-score
            logits = logits.sigmoid()
            pred_scores, pred_labels = logits.max(dim=1)
            pred_scores = logits[:, 1, :, :, :]

            label = label.reshape(-1).cpu()
            pred_labels = pred_labels.reshape(-1).cpu()
            pred_scores = pred_scores.reshape(-1).cpu()
            # 1.1 Cal confusiobn-matrix
            confu_mat = confusion_matrix(label, pred_labels)
            confusion_mat  = confusion_mat + confu_mat

            pred_scores = quant_scores(pred_scores, bin_size=0.001)
            # 1.2 Cal F-1 score
            try:
                F1 = f1_score(label, pred_labels)
                res['F1-Score'] += F1
                # # 2. AUC
                auc_score = roc_auc_score(label, pred_scores)
                res['AUC'] += auc_score 

                # 3. mAP
                precision, recall, thresholds = precision_recall_curve(label, pred_scores)
                AP = average_precision_score(label, pred_scores)
                res['mAP'] += AP 

                res['num_iters'] += 1
            except ValueError:
                print(label[:10])
            

            pred_labels_all = np.hstack([pred_labels_all, pred_labels])
            pred_scores_all = np.hstack([pred_scores_all, pred_scores])
            gt_labels_all   = np.hstack([gt_labels_all, label])
        
        # confusion_mat = confusion_matrix(gt_labels_all, pred_labels_all)
        # tn, fp, fn, tp = confusion_mat          # from official API 
        tn = confusion_mat[0][0]; fp = confusion_mat[0][1]
        fn = confusion_mat[1][0]; tp = confusion_mat[1][1]
        print("[Eval All]: \n", confusion_mat)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        res['pixel_nums'] = gt_labels_all.shape[0]
        res["recall"] = recall
        res['precision'] = precision

        res['F1-Score'] /= res['num_iters']
        res['mAP'] /= res['num_iters']
        res['AUC'] /= res['num_iters']

        print("Evaluation Results: ", res)

        return res


def valid(valid_dataloader,model ,criterion, aim_thres=0.51):
    with torch.no_grad():
        f1_list = []
        iou_list_6 = []
        iou_list_7 = []
        iou_list_8 = []
        iou_list_9 = []
        accuracy_list = []
        valid_loss_list = []
        auc_list = []
        logging.info('starting metrics')
        
        
        iterloader = iter(valid_dataloader)
        img, label = next(iterloader)
        img = img.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long)
        img = torch.reshape(img, (-1, 1, 64, 64, 64))
        logits = model(img)
        loss =criterion(logits, label)
        prob = torch.softmax(logits, dim=1)  # softmax on logits

        ones = torch.ones(img.shape).to(device)
        zeros = torch.zeros(img.shape).to(device)
        #predictions = torch.argmax(prob, dim=1).squeeze(dim=1)
        # from IPython import embed; embed()
        # predictions_6_scores, predictions_6_labels = prob.max(dim=1)
        # keep = predictions_6_scores > 0.5
        # predictions_6_labels = predictions_6_labels[keep]
        # predictions_6 = predictions_6_labels

        predictions_6 = torch.where(prob[:, 1:2, :, :, :] > aim_thres, ones, zeros).squeeze(dim=1)
        predictions_7 = torch.where(prob[:, 1:2, :, :, :] > 0.99, ones, zeros).squeeze(dim=1)
        predictions_8 = torch.where(prob[:, 1:2, :, :, :] > 0.999, ones, zeros).squeeze(dim=1)
        predictions_9 = torch.where(prob[:, 1:2, :, :, :] > 0.9999, ones, zeros).squeeze(dim=1)

        _, predictions_3_classes = prob.max(dim=1)
        predictions_3_classes = predictions_3_classes[:, 7:57, 7:57, 7:57]

        label = label[:, 7:57, 7:57, 7:57]
        prob = prob[:, 1:2, 7:57, 7:57, 7:57]
        
        predictions_6 = predictions_6[:, 7:57, 7:57, 7:57]
        predictions_7 = predictions_7[:, 7:57, 7:57, 7:57]
        predictions_8 = predictions_8[:, 7:57, 7:57, 7:57]
        predictions_9 = predictions_9[:, 7:57, 7:57, 7:57]
        
        all_class_mask = torch.where(torch.not_equal(img, 0), ones, zeros)
        
        all_class_mask = all_class_mask[:, :, 7:57, 7:57, 7:57].squeeze(dim=1)
        
        #auc = roc_auc_score(y_true = label.cpu().numpy().flatten(),y_score = prob.cpu().numpy().transpose(1,0,2,3,4).reshape(-1),multi_class = 'ovo',average = 'macro')
        auc = 0
        flattened_all_class_mask = torch.reshape(all_class_mask, (-1,))
        f1 = f1_score(label.cpu().numpy().flatten(), predictions_6.cpu().numpy().flatten(), average=None)

        conf_mat = confusion_matrix(label.cpu().numpy().flatten(), predictions_3_classes.cpu().numpy().flatten())
        print("conf_mat: \n", conf_mat)


        iou_6 = jaccard_score(label.cpu().numpy().flatten(), 
            predictions_6.cpu().numpy().flatten(), average=None,sample_weight =flattened_all_class_mask.cpu() )
        iou_7 = jaccard_score(label.cpu().numpy().flatten(), 
            predictions_7.cpu().numpy().flatten(), average=None,sample_weight =flattened_all_class_mask.cpu())
        iou_8 = jaccard_score(label.cpu().numpy().flatten(), 
            predictions_8.cpu().numpy().flatten(), average=None,sample_weight =flattened_all_class_mask.cpu())
        iou_9 = jaccard_score(label.cpu().numpy().flatten(), 
            predictions_9.cpu().numpy().flatten(), average=None,sample_weight =flattened_all_class_mask.cpu())

        accuracy = accuracy_score(label.cpu().numpy().flatten(), 
            predictions_6.cpu().numpy().flatten(), sample_weight=flattened_all_class_mask.cpu())
        #print(confusion_matrix(label.cpu().numpy().flatten(), predictions.cpu().numpy().flatten()))
        auc_list.append(auc)
        f1_list.append(f1)
        iou_list_6.append(iou_6)
        iou_list_7.append(iou_7)
        iou_list_8.append(iou_8)
        iou_list_9.append(iou_9)
        accuracy_list.append(accuracy)
        valid_loss_list.append(loss.item())
        
        logging.info('finish metrics')
        f1_list = np.array(f1_list)
        auc_list = np.array(auc_list)
        iou_list_6 = np.array(iou_list_6)
        iou_list_7 = np.array(iou_list_7)
        iou_list_8 = np.array(iou_list_8)
        iou_list_9 = np.array(iou_list_9)


        accuracy_list = np.array(accuracy_list)
        valid_loss_list = np.array(valid_loss_list)
        auc_mean = np.mean(auc_list, axis=0)
        f1_mean = np.mean(f1_list, axis=0)
       
        accuracy_mean = np.mean(accuracy_list, axis=0)
        valid_loss_mean = np.mean(valid_loss_list, axis=0)
        print('valid_f1 = {},valid_iou = {},{},{},{},valid_auc = {},valid_accuracy = {},valid_loss = {}'.format(
            f1_mean, 
            np.mean(iou_list_6, axis=0), np.mean(iou_list_7, axis=0),
            np.mean(iou_list_8, axis=0), np.mean(iou_list_9, axis=0),
            auc_mean, accuracy_mean, valid_loss_mean))

        val_res = {
            "valid_f1": f1_mean,
            "valid_iou": [np.mean(iou_list_6, axis=0), np.mean(iou_list_7, axis=0),
                          np.mean(iou_list_8, axis=0), np.mean(iou_list_9, axis=0),],
            "valid_auc": auc_mean,
            "valid_acc": accuracy_mean,
            "valid_loss": valid_loss_mean
        }
        return val_res


@torch.no_grad()
def _eval(args, aim_thres=0.51):
    valid_dataset = CaBackboneSet(index_csv='../datas/backbone_reducer/test.csv', label_mode=args.backbone_mode, out_channels=args.output_channel)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size,  num_workers=4, shuffle=False)

    model = UNet3D(
        in_channels=1, 
        out_channels=args.output_channel, 
        final_sigmoid=False, 
        f_maps=16, layer_order='cr',
        num_groups=8, 
        num_levels=5, 
        is_segmentation=True, conv_padding=1)
    
    if torch.cuda.device_count() > 1 :
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.load_from),False)
    model.eval()

    weight = torch.from_numpy(np.array(args.weight)).float()
    criterion = torch.nn.CrossEntropyLoss(weight = weight).to(device)

    val_res_1 = valid(valid_dataloader, model, criterion, aim_thres=0.51)
    val_res_2 = eval_all(valid_dataloader, model)

    print("-*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-")
    print("Eval-Res, with softmax and thres {}: ".format(aim_thres))
    print(val_res_1)
    print("###########################################################")
    print("Eval-Res, with max: ")
    print(val_res_2)
    print("-*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-")

def train(args):
    writer = SummaryWriter('./log')
    logging.info('loading dataset...')
    train_dataset = CaBackboneSet(index_csv='../datas/backbone_reducer/train.csv', label_mode=args.backbone_mode, out_channels=args.output_channel)
    valid_dataset = CaBackboneSet(index_csv='../datas/backbone_reducer/test.csv', label_mode=args.backbone_mode, out_channels=args.output_channel)
    logging.info('Finished')


    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size,  num_workers=4, shuffle=False)
    
    model = UNet3D(
        in_channels=1, 
        out_channels=args.output_channel, 
        final_sigmoid=False, 
        f_maps=16, layer_order='cr',
        num_groups=8, 
        num_levels=5, 
        is_segmentation=True, conv_padding=1)

    args.best_f1_score = 0.0
 
    if torch.cuda.device_count() > 1 :
        logging.info("Use "+str(torch.cuda.device_count())+' gpus')
        #print("Use", torch.cuda.device_count(), 'gpus')
        model = torch.nn.DataParallel(model)
    
    if len(args.resume_from) > 0:
        print("Loading params: ", args.resume_from)
        model.load_state_dict(
            torch.load(args.resume_from), strict=True,
        )

    weight = torch.from_numpy(np.array(args.weight)).float()
    criterion = torch.nn.CrossEntropyLoss(weight = weight).to(device)
    # criterion = FocalLoss(weight=torch.tensor([0.25, 0.75]), gamma=2).to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()

        for idx,(img, label) in enumerate(train_dataloader, 0):
            img = img.to(device,dtype=torch.float32)
            label = label.to(device,dtype=torch.long)
            img = torch.reshape(img,(-1, 1, 64, 64, 64))
            predictions  = model(img)
            optimizer.zero_grad()
            loss =criterion(predictions, label)
            loss.backward()
            optimizer.step()
            
            if idx % (len(train_dataloader) // 10)==0:
                logging.info("epoch={}/{},{}/{} of train, loss={}".format(
                    epoch, args.epochs, idx, len(train_dataloader),loss.item()))
        
        val_res = valid(valid_dataloader, model, criterion)
        if epoch % 2 == 0 and args.output_channel == 2:
            eval_res = eval_all(valid_dataloader, model)
            print(eval_res)

        # from IPython import embed; embed()
        val_f1 = val_res['valid_f1'][1]
        if val_f1 > args.best_f1_score:
            torch.save(model.state_dict(), './checkpoints/0112_0921_classes_{}_{}_{}_best_{}.pth'.format(
                args.output_channel, args.type, args.dataset, epoch
            ))
            torch.save(model.state_dict(), './checkpoints/0112_0921_' + 'classes_' + str(args.output_channel) + args.type+'_'+args.dataset+'_best_'+str(epoch) + '.pth')
            args.best_f1_score = val_f1
        # save model
        torch.save(model.state_dict(), './checkpoints/0112_0921_classes_{}_{}_{}_checkpoint_epoch_{}.pth'.format(
                args.output_channel, args.type, args.dataset, epoch
            ))       
    

if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='PyTorch Template')

    # parser.add_argument('--config', default='./unet_config.json', type=str)
    parser.add_argument('--config', default='./unet_config_class_3.json', type=str)
    
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=18, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--model', default='unet_model', type=str)
    parser.add_argument('--output_channel', default=2, type=int)
    parser.add_argument('--weight', default=[0.01,1] ,help = 'crossentropy loss weight')
    parser.add_argument('--type', default='backbone', type=str)
    parser.add_argument('--dataset', default='simulation', type=str)
    parser.add_argument('--phase', default='train', type=str)
    #parser.add_argument('--use_multigpus', default='True', action="store_true")
    
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        args.__dict__ = json.load(f)
    logging.info('args: ' + str(args) + '...')

    train(args) 