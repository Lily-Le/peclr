#%%
import matplotlib.pyplot as plt
import numpy as np
import os

labels=[]
BASE_PATH1='/home/zlc/cll/code/peclr_cbg/'

base1='finetune_result/hybrid2-frei-cgbg-correct/epoch_14/'
base2='finetune_result/hybrid2-frei-cgbg-correct/epoch_29/'

save_base1=os.path.join(BASE_PATH1,base1)
save_base2=os.path.join(BASE_PATH1,base2)

labels.append(['cgbg_ep14','cgbg_ep29'])

BASE_PATH2='/home/zlc/cll/code/peclr/'
base3='finetune_result/models_res18/hybrid2-ori-data4/'
# save_base='/home/zlc/cll/code/peclr_cbg/data/models/finetune/npretrained_trainAll/'
base4='finetune_result/res18_trall/'

save_base3=os.path.join(BASE_PATH2,base3)
save_base4=os.path.join(BASE_PATH2,base4)

labels.append(['ori_ep14','res18_trainAll'])
# save_base5='/home/zlc/cll/code/contra-hand/finetune/res50_0175/'
# save_base1,save_base3,
# 'peclr_bs256',,save_base5
save_bases=[save_base1,save_base2,save_base3,save_base4]
# labels=[ 'peclr_bs256','peclr_imgnet','peclr_change_bg','peclr','moco_hanco']
# labels=['peclr_imgnet','peclr']

epochs=1000
x = range(epochs)
accs=[]
val_accs=[]
losses=[]
val_losses=[]
fig=plt.figure(figsize=(10,10))
ax_acc=fig.add_subplot(220+1)
ax_val=fig.add_subplot(220+1)
axes=[]
#%%
# acc acc_val, loss, loss_val
for i in range(4):
    axes.append(fig.add_subplot(220+1))
    axes[i].set_xlabel('Epoch')

i=0
ep=46
for n in save_bases:#_ep999_
    # accs.append(np.load(n+f'Accuracy_list_ep{ep}.npy',allow_pickle=True))
    # val_accs.append(np.load(n+f'Accuracy_val_list_ep{ep}.npy',allow_pickle=True))
    # losses.append(np.load(n+f'Loss_list_ep{ep}.npy',allow_pickle=True))
    # val_losses.append(np.load(n+f'Loss_val_list_ep{ep}.npy',allow_pickle=True))
    # accs=[]
    # val_accs=[]
    # losses=[]
    # val_losses=[]
    tmp=np.load(n+f'Accuracy_list_ep{ep}.npy',allow_pickle=True)
    tmp2=[]
    for j in range(len(tmp)):
        tmp2.append(tmp[j].cpu().numpy())
    np.save(n+f'Accuracy_list_ep{ep}n.npy',tmp2)
    # accs.append(np.load(n+f'Accuracy_list_ep{ep}.npy',allow_pickle=True))
    
    # val_accs.append(np.load(n+f'Accuracy_val_list_ep{ep}.npy',allow_pickle=True))
    # losses.append(np.load(n+f'Loss_list_ep{ep}.npy',allow_pickle=True))
    # val_losses.append(np.load(n+f'Loss_val_list_ep{ep}.npy',allow_pickle=True))
    
    # axes[0].plot(tmp,label=labels[i])
    # axes[1].plot(val_accs[-1],label=labels[i])
    # axes[2].plot(losses[-1],label=labels[i])
    # axes[3].plot(val_losses[-1],label=labels[i])
    # break
    i+=1


#%%
# ax.plot(Accuracy_val_list, label = 'val_acc')

axes[0].set_ylabel('Accuracy')
# axes[1].set_ylabel('Val_Accuracy')
# axes[2].set_ylabel('Loss')
# axes[3].set_ylabel('Val_Loss')
#plt.ylim([0.5, 1])
axes[0].set_title('Accuracy vs. epoches')
plt.legend(loc='lower right')

# 

# %%
