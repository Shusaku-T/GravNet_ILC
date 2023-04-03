import os, os.path as osp
from time import strftime
import tqdm
import torch
from torch_geometric.loader import DataLoader
import argparse

import matplotlib.pylab as plt 
import numpy as np

from sklearn.metrics import accuracy_score
import torch_cmspepr.objectcondensation as oc
import torch.nn.functional as f

from gravnet_model import GravnetModel,GravnetModelWithNoiseFilter
from dataset_ILC import ILCDataset
from lrscheduler import CyclicLRWithRestarts
from sklearn.manifold import TSNE

from ReadText import ReadText
import sys

torch.manual_seed(1009)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dry', action='store_true', help='Turn off checkpoint saving and run limited number of events')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more output')
    parser.add_argument('--settings-Sep01', action='store_true', help='Use 21Sep01 settings')
    parser.add_argument('--reduce-noise', action='store_true', help='Randomly kills 95% of noise')
    parser.add_argument('--ckptdir', type=str)
    args = parser.parse_args()
    if args.verbose: oc.DEBUG = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    reduce_noise = True

    n_epochs = 15
    batch_size = 100

    shuffle = True

    dataset = ILCDataset(ReadText("Grav_ILC_setting3.txt")["Data Folder"])
    
    if reduce_noise:
        dataset.reduce_noise = .70
        multiply_batch_size = 1
        print(f'Throwing away {dataset.reduce_noise*100:.0f}% of noise (good for testing ideas, not for final results)')
        print(f'Batch size: {batch_size} --> {multiply_batch_size*batch_size}')
        batch_size *= multiply_batch_size
    if args.dry:
        keep = .005
        print(f'Keeping only {100.*keep:.1f}% of events for debugging')
        dataset, _ = dataset.split(keep)
    
    train_dataset, test_dataset = dataset.split(.8)
    print(f"train_dataset :{len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    
    model = GravnetModel(input_dim=4, output_dim=3).to(device)
    #else:
    #    model = GravnetModel(input_dim=4, output_dim=3, k=50).to(device)

    # Checkpoint loading
    # if True:
    #     # ckpt = 'ckpts_gravnet_Aug27_2144/ckpt_9.pth.tar'
    #     ckpt = 'ckpts_gravnet_Aug27_0502/ckpt_23.pth.tar'
    #     print(f'Loading initial weights from ckpt {ckpt}')
    #     model.load_state_dict(torch.load(ckpt, map_location=device)['model'])

    epoch_size = len(train_loader.dataset)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=9.0e-6, weight_decay=1e-4)

    if not args.settings_Sep01:
        scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")

    loss_offset =1. # To prevent a negative loss from ever occuring

    train_accu=[]
    test_accu=[]

    # def loss_fn(out, data, s_c=1., return_components=False):
    #     device = out.device
    #     pred_betas = torch.sigmoid(out[:,0])
    #     pred_cluster_space_coords = out[:,1:]
    #     assert all(t.device == device for t in [
    #         pred_betas, pred_cluster_space_coords, data.y, data.batch,
    #         ])
    #     out_oc = oc.calc_LV_Lbeta(
    #         pred_betas,
    #         pred_cluster_space_coords,
    #         data.y.long(),
    #         data.batch,
    #         return_components=return_components
    #         )
    #     if return_components:
    #         return out_oc
    #     else:
    #         LV, Lbeta = out_oc
    #         return LV + Lbeta + loss_offset

    def check_coords(out,data) :
        learning_para={}
        #pred_betas = torch.sigmoid(out[:,0])
        pred_cluster_space_coords = out[:,1:]
        #learning_para["pred_betas"] =pred_betas
        learning_para["pred_cluster_space_coords"] =pred_cluster_space_coords
        #print(f"coords_test_shape:{pred_cluster_space_coords.shape}")
        learning_para["data.y.long"]=data.y.long()
        learning_para["data.batch"] = data.batch
        return learning_para

    def check_data(data):
        data_para={}
        data_para["data.y.long"]=data.y.long()
        data_para["data.x"]=data.x
        return data_para
        
    def loss_fn(out, data, i_epoch=None, return_components=False):
        device = out.device
        pred_betas = torch.sigmoid(out[:,0])
        pred_cluster_space_coords = out[:,1:]
        assert all(t.device == device for t in [
            pred_betas, pred_cluster_space_coords, data.y, data.batch,
            ])
        out_oc = oc.calc_LV_Lbeta(
            pred_betas,
            pred_cluster_space_coords,
            data.y.long(),
            data.batch,
            return_components=return_components,
            beta_term_option='short-range-potential',
            )
        if return_components:
            return out_oc
        else:
            LV, Lbeta = out_oc
            if i_epoch <= 7:
                return LV + loss_offset
            else:
                return LV + Lbeta + loss_offset

    def train(epoch):
        print('Training epoch', epoch)
        train_acc=0.
        cluster_space_coords_list=[]
        data_y_list=[]
        model.train()
        if not args.settings_Sep01: scheduler.step()
        try:
            pbar = tqdm.tqdm(train_loader, total=len(train_loader))
            pbar.set_postfix({'loss': '?'})
            for i, data in enumerate(pbar):
                data = data.to(device)
                optimizer.zero_grad()
                if i == 0 : first_para = check_data(data)
                result = model(data.x, data.batch)
                learning_para = check_coords(result,data)
                loss = loss_fn(result, data, i_epoch=epoch)
                loss.backward()
                optimizer.step()
                if not args.settings_Sep01: scheduler.batch_step()
                pbar.set_postfix({'loss': float(loss)})
                cluster_space_coords_list.append(learning_para["pred_cluster_space_coords"].tolist())
                data_y_list.append(learning_para["data.y.long"].tolist())
                # if i == 2: raise Exception
            return loss.item(),cluster_space_coords_list,data_y_list,data,first_para
        except Exception:
            print('Exception encountered:', data, ', npzs:')
            print('  ' + '\n  '.join([train_dataset.npzs[int(i)] for i in data.inpz]))
            raise
    
        
    def test(epoch):
        N_test = len(test_loader)
        loss_components = {}
        test_acc=0.
        def update(components):
            for key, value in components.items():
                if not key in loss_components: loss_components[key] = 0.
                loss_components[key] += value
        with torch.no_grad():
            
            model.eval()
            for data in tqdm.tqdm(test_loader, total=len(test_loader)):
                data = data.to(device)
                result = model(data.x, data.batch)
                update(loss_fn(result, data, return_components=True))
        # Divide by number of entries
        for key in loss_components:
            loss_components[key] /= N_test
        # Compute total loss and do printout
        print('test ' + oc.formatted_loss_components_string(loss_components))
        test_loss = loss_offset + loss_components['L_V']+loss_components['L_beta']
        print(f'Returning {test_loss}')
        return test_loss.item()

    ckpt_dir = strftime('checkpoint/ckpts_gravnet_Momentum15_5_%b%d_%H%M') if args.ckptdir is None else args.ckptdir
    def write_checkpoint(checkpoint_number=None, best=False):
        ckpt = 'ckpt_best.pth.tar' if best else 'ckpt_{0}_1.pth.tar'.format(checkpoint_number)
        ckpt = osp.join(ckpt_dir, ckpt)
        if best: print('Saving epoch {0} as new best'.format(checkpoint_number))
        if not args.dry:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(dict(model=model.state_dict()), ckpt)

    min_loss = 1e9
    train_loss_history=[]
    test_loss_history=[]
    epoch_history=[]
    train_acc_history=[]
    test_acc_history=[]
    for i_epoch in range(n_epochs):
        train_loss,cluster_space_para,data_y,data,first_para=train(i_epoch)
        train_loss_history.append(train_loss)
        write_checkpoint(i_epoch)
        
        test_loss= test(i_epoch)
        #test_loss/=len(test_loader)
        test_loss_history.append(test_loss)
        if test_loss < min_loss:
            min_loss = test_loss
            #write_checkpoint(i_epoch, best=True)

        #if i_epoch==0 or i_epoch==30 : check_plots(cluster_space_para,data_y)
        #if i_epoch==30 : check_plots(cluster_space_para,data_y)
        
    data_y = data.y.long().cpu().numpy()
    plot_history(train_loss_history,test_loss_history)

def colorlabel(y,label):
    unique_label=np.unique(label)
    if y == unique_label[0] :
        return "b"
    elif y == unique_label[1] :return "g"

def check_plots(coords_list,data_y_list):
    #coords_lists has the diferent numbers of elements for each row, so it cannot be converted to numpy!!!!!!!!!!!!!!!!!!!!!
    coords_list=np.array(coords_list[0])
    label=np.array(data_y_list[0][0:4000])
    fig,ax = plt.subplots(figsize = (8,6))
    l = 0
    for x1,y1,label1 in zip(coords_list[0:4000,0],coords_list[0:4000,1],label):
        ax.scatter(x1, y1,c=colorlabel(label1,label))
    plt.show()
    
def coord_tsne(Coords,Tag):
    tsne = TSNE(n_components=2,random_state=41,learning_rate='auto')
    Coord_reduced = tsne.fit_transform(Coords)

    plt.figure(figsize=(13,7))
    plt.scatter(Coord_reduced[0:4000,0],Coord_reduced[0:4000,1],c=Tag,cmap='jet',s=15,alpha=0.5)
    #plt.axis('off')
    plt.colorbar()
    plt.show()

def debug():
    oc.DEBUG = True
    dataset = TauDataset('data/taus')
    dataset.npzs = [
        # 'data/taus/49_nanoML_84.npz',
        # 'data/taus/37_nanoML_4.npz',
        #'data/taus/26_nanoML_93.npz',
        # 'data/taus/142_nanoML_75.npz',
        ]
    for data in DataLoader(dataset, batch_size=len(dataset), shuffle=False): break
    print(data.y.sum())
    model = GravnetModel(input_dim=9, output_dim=4)
    with torch.no_grad():
        model.eval()
        out = model(data.x, data.batch)
    pred_betas = torch.sigmoid(out[:,0])
    pred_cluster_space_coords = out[:,1:4]
    out_oc = oc.calc_LV_Lbeta(
        pred_betas,
        pred_cluster_space_coords,
        data.y.long(),
        data.batch.long()
    )
        
def plot_history(train_loss_history,test_loss_history):
    loss_type = type(test_loss_history)
    if(loss_type is list):
        plt.figure(figsize=(8,6))
        plt.plot(test_loss_history,label='test_loss', lw=3, c='b')
        plt.plot(train_loss_history,label='train_loss',lw=3,c='green')
        plt.title('loss function')
        plt.legend(fontsize=14)
        plt.show()

def plot_acc_history(train_acc_history,test_acc_history):
    loss_type = type(test_acc_history)
    if(loss_type is list):
        plt.figure(figsize=(8,6))
        plt.plot(test_acc_history,label='test_acc', lw=3, c='b')
        plt.plot(train_acc_history,label='train_acc',lw=3,c='green')
        plt.title('accuracy')
        plt.legend(fontsize=14)
        plt.show()
            
def run_profile():
    from torch.profiler import profile, record_function, ProfilerActivity
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    batch_size = 2
    n_batches = 2
    shuffle = True
    dataset = TauDataset('data/taus')
    dataset.npzs = dataset.npzs[:batch_size*n_batches]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f'Running profiling for {len(dataset)} events, batch_size={batch_size}, {len(loader)} batches')

    model = GravnetModel(input_dim=9, output_dim=8).to(device)
    epoch_size = len(loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=1e-4)

    print('Start limited training loop')
    model.train()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            pbar = tqdm.tqdm(loader, total=len(loader))
            pbar.set_postfix({'loss': '?'})
            for i, data in enumerate(pbar):
                data = data.to(device)
                optimizer.zero_grad()
                result = model(data.x, data.batch)
                loss = loss_fn(result, data)
                print(f'loss={float(loss)}')
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': float(loss)})
    print(prof.key_averages().table(sort_by="cpu_time", row_limit=10))
    # Other valid keys:
    # cpu_time, cuda_time, cpu_time_total, cuda_time_total, cpu_memory_usage,
    # cuda_memory_usage, self_cpu_memory_usage, self_cuda_memory_usage, count

if __name__ == '__main__':
    pass
    main()
    # debug()
    # run_profile()
