import evaluation_noNoise as ev
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import sys
import event_view_plot as plt_3d
#from importlib import reload
#import mplhep

def event_count(event,clustering,count):
    true_cluster_num = np.unique(event.y).size
    predicted_cluster_num = np.unique(clustering).size
    count[0].append(true_cluster_num)
    count[1].append(predicted_cluster_num)
    return count

def count_hit_inCluster(event,clustering,acc):
    count = check_prediction(event,clustering)
    #assert count in range(len(count))
    #assert sorted(count[0].values(),reverse=True) in range(len(sorted(count[0].values(),reverse=True)))

    if len(count[0]) == 1 or len(count[1]) == 1 : return acc
    accuracy_1stcluster = sorted(count[1].values(),reverse=True)[0] / sorted(count[0].values(),reverse=True)[0]
    if len(sorted(count[1].values(),reverse=True))!=1 and len(sorted(count[0].values(),reverse=True)) !=1:       
        accuracy_2ndcluster = sorted(count[1].values(),reverse=True)[1] / sorted(count[0].values(),reverse=True)[1]
    else : plt_3d.event_display(event,clustering) 
    acc[0].append(accuracy_1stcluster)
    #acc[1].append(accuracy_2ndcluster)
    return acc

def check_prediction(event,clustering):
    count=[{},{}]
    type_hit_true = np.unique(event.y)
    type_hit_predicted = np.unique(clustering)
    for type_hit in type_hit_true:
        count[0][type_hit]=np.count_nonzero(event.y == type_hit) #count the number of hits of true label
    for type_hit in type_hit_predicted:
        count[1][type_hit]=np.count_nonzero(clustering == type_hit) #predicted label
    return count
    
def ilclabel(ax, text=r'$\itSimulation \; Preliminary$', x=.12, y=.88, dx=.12):
    ax.text(x, y, r'$\bfILD$', ha='left', va='bottom',transform=ax.figure.transFigure,fontsize=36)
    ax.text(x+dx, y, text,ha='left', va='bottom',transform=ax.figure.transFigure,fontsize=28)

def plot_stats(stats_numpy,hist_min,hist_max,key_name,num):
    fig = plt.figure(facecolor="white",figsize=[9,5])
    ax = fig.gca()
    #ax.hist(stats['eta_truth'])
    bins = np.linspace(hist_min, hist_max, 50)
    #epred_o_etruth = stats['n_pred_id'][sel] / stats['n_truth_id'][sel]
    ax.hist(stats_numpy, bins=bins, linewidth=2)
    average_hist = np.round(np.nanmean(stats_numpy),decimals=5)
    #average_hist = np.mean(stats_numpy)
    ax.text(0.85,0.85,f"average : {average_hist}")
    plt.show()
    #fig.savefig(f"GravPNG/{key_name}_{hist_min}_{hist_max}_{num}.png")
    
def plot_n_pred_in_truth(stats):
    fig = plt.figure(facecolor="white",figsize=[9,5])
    ax = fig.gca()
    #ax.hist(stats['eta_truth'])
    cat_name = {0: 'EM', 1: 'HAD', 2: 'MIP', 3: 'MIX'}

    ax.set_xlabel(r'$\Sigma E_{hit}^{pred} / \Sigma E_{hit}^{truth}$')
    ax.set_xlabel(r'$N_{hits}^{pred} / N_{hits}^{truth}$ (energy weighted)')
    ax.set_ylabel('A.U.')
    bins = np.linspace(0., 1.2, 50)
    
    #epred_o_etruth = stats['n_pred_id'][sel] / stats['n_truth_id'][sel]
    epred_o_etruth = stats['rate_pred']
    ax.hist(epred_o_etruth, bins=bins, linewidth=2)
    plt.show()

def scan_stats_plt_eachEvent(tbeta=.2,td=.5,nmax=4,yielder=None):
    stats = ev.Stats()
    if yielder is None :yielder = ev.TestYielder()
    count_cluster_inEvent=[[],[]]
    rate_accuracy = [[],[]]
    energy_rate_accuracy=[[],[]]
    for event,prediction,clustering,matches in yielder.iter_matches(tbeta,td,nmax):
        if(len(event.y)<10):continue
        check = ev.get_hit_matched_vs_unmatched_noStat_eachEvent(event,clustering,matches)
        if check == "y" : break
        elif check == "z" : continue
    
def scan_stats_plt(tbeta=.2,td=.5,nmax=4,yielder=None):
    stats = ev.Stats()
    if yielder is None :yielder = ev.TestYielder()
    count_cluster_inEvent=[[],[]]
    rate_accuracy = [[],[]]
    energy_rate_accuracy=[[],[]]
    for event,prediction,clustering,matches in yielder.iter_matches(tbeta,td,nmax):
        if(len(event.y)<10):continue
        check = ev.get_hit_matched_vs_unmatched_noStat(event,clustering,matches)
        if check == "y" : break
        elif check == "z" : continue

def scan_stats(tbeta=.2,td=.5,nmax=4,yielder=None):
    stats = ev.Stats()
    if yielder is None :yielder = ev.TestYielder()
    count_cluster_inEvent=[[],[]]
    rate_accuracy = [[],[]]
    energy_rate_accuracy=[[],[]]
    check = True
    for event,prediction,clustering,matches in yielder.iter_matches(tbeta,td,nmax):
        if(len(event.y)<10):continue
        #count_cluster_inEvent=event_count(event,clustering,count_cluster_inEvent)
        #rate_accuracy=count_hit_inCluster(event,clustering,rate_accuracy)
        #energy_rate_accuracy,rate_1st,rate_2nd=cal_energy_rate(event,clustering,energy_rate_accuracy)
        #if rate_1st <0.1:
        #    plt_3d.event_display(event,clustering)

def get_stats_test(tbeta=.2, td=.5, nmax=4, yielder=None):
    stats = ev.Stats()
    yielder = ev.TestYielder()
    for event, prediction, clustering,matches in yielder.iter_matches(tbeta,td,nmax):
        print(f"event : {type(event.y)}")
        print(f"clustering : {type(clustering)}")
        print(f"matches : {type(matches[0][0])}")
        

def get_stats(tbeta=.2, td=.5, nmax=4, yielder=None,accType="number"):
    stats = ev.Stats()
    if yielder is None:yielder = ev.TestYielder()
    count_cluster_inEvent = [[],[]]
    rate_accuracy = [[],[]]
    energy_rate_accuracy = [[],[]]
    check = True
    for event, prediction, clustering,matches in yielder.iter_matches(tbeta,td,nmax):
        #count_cluster_inEvent=event_count(event,clustering,count_cluster_inEvent)
        #rate_accuracy=count_hit_inCluster(event,clustering,rate_accuracy)
        #energy_rate_accuracy,rate_1st,rate_2nd = cal_energy_rate(event,clustering,energy_rate_accuracy)
        #if check == True :
        #    print(f"matches : {matches}")
        #    if len(matches[1]) > 2:check = False
        #stats.extend(ev.statistics_per_match(event,clustering,matches))
        if accType == "number":
            stats.extend(ev.get_hit_matched_vs_unmatched(event,clustering,matches))
        elif accType =="energy":
            stats.extend(ev.get_hit_matched_vs_unmatched_energy(event,clustering,matches))
        #stats.extend(ev.show_Stats(event,clustering,matches))
        #stats.add('confmat',ev.signal_to_noise_confusion_matrix(event,clustering,norm=True))
    return stats

def show_accNumber():
    stats=get_stats(nmax=20000,accType="number")
    plot_stats(stats['num_pred'],0,1,"num_pred",1)
    plot_stats(stats['rate_pred'],0,1,"rate_pred",1)

def show_accEnergy():
    stats=get_stats(nmax=20000,accType="energy")
    plot_stats(stats['num_pred_energy'],0,1,"num_pred_energy",1)
    plot_stats(stats['rate_pred_energy'],0.99,1,"rate_pred_energy",1)

def check_match():
    path="/home/tsumura/ILC/GravNet/pytorch_ILC_PFA/check_Grav/grav_check2.txt"
    if os.path.isfile(path) :
        with open(path,'w') as f:
            f.write('')
    stats = get_stats(nmax=20000)
    check = True
    for matched_pred,unmatched_pred in zip(stats['matched_pred_size'],stats['unmatched_pred_size']):
        for matched_pred in stats['matched_pred']:
            if check:
                print(f"matched pred :{matched_pred}")
                print(f"matched pred size :{matched_pred}")
                print(f"unmatched pred size :{unmatched_pred}")
                check = False
    print(f"hit unmatch pred : {stats['nhits_unmatched_pred'][0]}")
    ev.dump_stats('stats_%b%d',stats)
    nhit_energy_acc(stats)
    plot_stats(stats['rate_pred'],0.9,1.,"rate_pred",sys.argv[1])
    plot_stats(stats['rate_pred'],0.,1.,"rate_pred",sys.argv[1])
    plot_n_pred_in_truth(stats)
    nhit_truth(stats)

def check_scan_stats():
    stats=scan_stats(nmax=5)
    nhit_acc(stats)
    plot_stats(stats['sum_truth'])
    plot_n_pred_in_truth(stats)

def main():
    show_accNumber()
    #show_accEnergy()

if __name__=='__main__':
    main()
    
