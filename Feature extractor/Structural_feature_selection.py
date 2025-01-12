from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
from sklearn.model_selection import KFold
import torch
from tqdm import tqdm
import os
from himalaya.lasso import SparseGroupLassoCV
from himalaya.backend import set_backend

os.environ['CUDA_VISIBLE_DEVICES'] = "4"
backend = set_backend("torch_cuda", on_error="warn")
scaler = StandardScaler()
ROIs = ['VO','V1','V2','V3','V3ab','PHC','MT','MST','LO','IPS','hV4']

def fetch_ROI_voxel(file_ex,ROIs):
    file_paths = [file_ex+roi+'.npy' for roi in ROIs]
    ROI_voxel_value = np.concatenate([np.load(file_paths[i]) for i in range(len(file_paths))], axis=1)
    return ROI_voxel_value

def fetch_CLIP_feature(file_name):
    CLIP_feature_all_layer = np.load(file_name)
    return CLIP_feature_all_layer

def CLIP_feature_layer(CLIP_feature_all_layer,layer_name):
    index = int((int(layer_name[-1])+1)/2)
    CLIP_layer = CLIP_feature_all_layer[:, 153600 * (index - 1):153600 * index]
    return CLIP_layer

def cal_pccs(X, Y):
	XMean = np.mean(X)
	YMean = np.mean(Y)
	
	XSD = np.std(X)
	YSD = np.std(Y)
	
	ZX = (X-XMean)/(XSD+1e-30)
	ZY = (Y-YMean)/(YSD+1e-30)
	r = np.sum(ZX*ZY)/(len(X))
	return(r)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def feature_select_onefold(x_trn, k_folder, rate, fold_id, y_trn_all):
    x_trn = x_trn
    y_trn = y_trn_all[:, 38400 * (fold_id - 1):38400 * fold_id]
    kf = KFold(n_splits=k_folder, shuffle=False)
    cor_k_folder = []
    for train_index, test_index in kf.split(x_trn):
        x_fold1_train_data, x_fold1_test_data = x_trn[train_index], x_trn[test_index]
        y_fold1_train_data, y_fold1_test_data = y_trn[train_index], y_trn[test_index]
        model_him = SparseGroupLassoCV()
        model_him.fit(x_fold1_train_data, y_fold1_train_data)
        pred = model_him.predict(x_fold1_test_data).cpu().detach().numpy()

        cor = []
        for i in tqdm(range(pred.shape[1])):
            x = y_fold1_test_data[:, i]
            y = pred[:, i]
            cor_i = cal_pccs(x, y)
            # cor_i = scipy.stats.pearsonr(x,y)[0]
            cor.append(abs(cor_i))
        cor_k_folder.append(cor)
        print("-------------------------------------------------------------------")
    cor_mean = np.mean(np.array(cor_k_folder), axis=0)
    return cor_mean


def feature_select(index_save_path, CLIP_feature_saved_path, x_trn, k_folder=5, layer_name = 'Linear-1',rate = 50):
    index_save_path = index_save_path + '/{}/'.format(layer_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    trn_CLIP_feature_all_layer = np.concatenate([fetch_CLIP_feature(CLIP_feature_saved_path + "/trn_stim_CLIP_zscore_{}.npy".format(i + 1)) for i in range(1,9)], axis=0)
    y_trn_all = CLIP_feature_layer(trn_CLIP_feature_all_layer, layer_name)
    cor_all = np.concatenate([feature_select_onefold(x_trn = x_trn, k_folder=k_folder, rate=rate, fold_id=i+1, y_trn_all=y_trn_all) for i in range(4)] , axis = 0)
    bar = np.percentile(cor_all, rate)
    selected_index = []
    for i in range(cor_all.shape[0]):
        if cor_all[i] >= bar:
            selected_index.append(i)
    np.save(index_save_path + 'Folder_5_{}.npy'.format(100 - rate), np.array(selected_index))
    return


def main():
    parser = argparse.ArgumentParser(description='CLIP text feature extraction')
    parser.add_argument('--trn_fMRI_file_ex',  default='', type=str)
    parser.add_argument('--index_save_path',  default='', type=str)
    parser.add_argument('--CLIP_feature_saved_path',  default='', type=str)
    parser.add_argument('--Layers',  default=['Linear-2', 'Linear-4', 'Linear-6', 'Linear-8', 'Linear-10', 'Linear-12'], type=list)
    parser.add_argument('--rate',  default=20, type=float)
    args = parser.parse_args()
	
    x_trn = fetch_ROI_voxel(args.trn_file_ex, ROIs)  # (8859,11694)
    x_trn = scaler.fit_transform(x_trn)
    for layer in args.Layers:
	b = feature_select(index_save_path=args.index_save_path, CLIP_feature_saved_path=args.CLIP_feature_saved_path, x_trn=x_trn, k_folder=5, layer_name = 'Linear-1',rate = args.rate)
	    

if __name__ == "__main__":
    main()
	




