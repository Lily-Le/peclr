import torch
import torchvision
import torch
from tqdm import tqdm
import os

def peclr_to_torchvision(resnet_model, path_to_peclr_weights):
    """Copies parameters from trained peclr model to a corresponding resent from torchvision.
    All the weights until the fc layer of the resent are copied.
    
    Note: Make sure the saved PeCLR model shares same resent size as the "resnet_model" 

    Args:
        resnet_model (tochvision.models.ResNet): A resnet model
        path_to_peclr_weights (str)

    Raises:
        Exception: When the passed model is not of ResNet type
    """
    peclr_weights = torch.load(path_to_peclr_weights, map_location=torch.device("cpu"))
    print(peclr_weights.keys())
    peclr_state_dict = peclr_weights["state_dict"]
    if isinstance(resnet_model, torchvision.models.ResNet):
        resnet_state_dict_list = list(resnet_model.state_dict().items())
        peclr_state_dict_list = [
            (key, peclr_state_dict[key])
            for key in peclr_state_dict
            if "features" in key
        ]
        last_feature_idx = len(peclr_state_dict_list)
        own_state = resnet_model.state_dict()
        for idx in tqdm(range(last_feature_idx)):
            if (
                resnet_state_dict_list[idx][0].split(".")[-1]
                != peclr_state_dict_list[idx][0].split(".")[-1]
            ):
                print("PeCLR layers don't match with Resnet layer ")
                break
            name = resnet_state_dict_list[idx][0]
            param = peclr_state_dict_list[idx][1]
            try:
                own_state[name].copy_(param)
            except Exception as e:
                print("The models are not compatible!")
                print(f"Exception :{e}")
                break
    else:
        raise Exception(f"The selected model is not of type ResNet from torch vision!")
    return own_state

def main2():
    # resnet152 = torchvision.models.resnet152(pretrained=True)
    # # NOTE: This path is just for demonstration.
    # peclr_to_torchvision(resnet152,"data/models/637ab5816c43483181143d6189cc39ff/checkpoints/epoch=3.ckpt")
    
    resnet50 = torchvision.models.resnet18(pretrained=True)
    # resnet50_2 = torchvision.models.resnet18(pretrained=True)
    # NOTE: This path is just for demonstration.
    base_path = '/home/zlc/cll/code/peclr_cbg/data/models_res18/hybrid2-frei-cgbg/e6552f34152142bebf9f79184b2f85aa/'
    # base_path = '/workspace/cll/Exp_results/peclr/models_res50/hybrid2-frei-cgbg-correct/9d0a339319674ec980f1645a8e50de3d/'
    
    epoch=29
    checkpoint = f'checkpoints/epoch={epoch}.ckpt'
    model_path=os.path.join(base_path,checkpoint)
    save_path0 = os.path.join(base_path,'port_model')
    if not(os.path.exists(save_path0)):
        os.mkdir(save_path0)
    # model_path = '/home/zlc/cll/code/peclr_cbg/data/models/hybrid2-frei-bs256/epoch=286.ckpt'
    # model_path = '/home/zlc/cll/code/peclr/data/models/hybrid2-frei-cgbgr/566c9864e9e648fab0011ee69b9fa255/epoch=296.ckpt'
    # model_path = '/home/zlc/cll/code/peclr_cbg/data/models_res18/hybrid2-frei-cgbg-correct/66e551fefbd14447ab967161e9af9c0c/checkpoints/epoch=29.ckpt'
    # model_path = '/home/zlc/cll/code/peclr/data/models_res18/hybrid2-ori-data4/7992e56f43e34e75b3771644670c5898/checkpoints/epoch=29.ckpt'
    # model_path='/home/zlc/cll/code/peclr/data/models_res18/hybrid2-ori-data4/7992e56f43e34e75b3771644670c5898/checkpoints/epoch=44.ckpt'

    state_dic=peclr_to_torchvision(resnet50,model_path)
    save_path = os.path.join(base_path,f'port_model/epoch={epoch}.pth')
    ###save_path = os.path.join('/workspace/cll/Exp_results/peclr/pretrained_res18_state.pth')

    # resnet50_2.load_state_dict(state_dic)
    # save_path = '/home/zlc/cll/code/peclr/data/models_res18/hybrid2-ori-data4/7992e56f43e34e75b3771644670c5898/port_model/'
    
    torch.save(resnet50,save_path)
    statedic_path=os.path.join(base_path,f'port_model/epoch={epoch}_state.pth')
    torch.save(resnet50.state_dict(),statedic_path)


def port_ori():
    resnet50 = torchvision.models.resnet18(pretrained=True)
    # resnet50_2 = torchvision.models.resnet18(pretrained=True)
    # NOTE: This path is just for demonstration.
    # base_path = '/home/zlc/cll/code/peclr_cbg/data/models_res18/hybrid2-frei-cgbg/e6552f34152142bebf9f79184b2f85aa/'
    # base_path = '/workspace/cll/Exp_results/peclr/models_res50/hybrid2-frei-cgbg-correct/9d0a339319674ec980f1645a8e50de3d/'
    
    # epoch=29
    # checkpoint = f'checkpoints/epoch={epoch}.ckpt'
    # model_path=os.path.join(base_path,checkpoint)
    # save_path0 = os.path.join(base_path,'port_model')
    # if not(os.path.exists(save_path0)):
    #     os.mkdir(save_path0)
    # # model_path = '/home/zlc/cll/code/peclr_cbg/data/models/hybrid2-frei-bs256/epoch=286.ckpt'
    # # model_path = '/home/zlc/cll/code/peclr/data/models/hybrid2-frei-cgbgr/566c9864e9e648fab0011ee69b9fa255/epoch=296.ckpt'
    # # model_path = '/home/zlc/cll/code/peclr_cbg/data/models_res18/hybrid2-frei-cgbg-correct/66e551fefbd14447ab967161e9af9c0c/checkpoints/epoch=29.ckpt'
    # # model_path = '/home/zlc/cll/code/peclr/data/models_res18/hybrid2-ori-data4/7992e56f43e34e75b3771644670c5898/checkpoints/epoch=29.ckpt'
    # # model_path='/home/zlc/cll/code/peclr/data/models_res18/hybrid2-ori-data4/7992e56f43e34e75b3771644670c5898/checkpoints/epoch=44.ckpt'

    # state_dic=peclr_to_torchvision(resnet50,model_path)
    # save_path = os.path.join(base_path,f'port_model/epoch={epoch}.pth')
    save_path = os.path.join('/workspace/cll/Exp_results/peclr/pretrained_res18_state.pth')

    # resnet50_2.load_state_dict(state_dic)
    # save_path = '/home/zlc/cll/code/peclr/data/models_res18/hybrid2-ori-data4/7992e56f43e34e75b3771644670c5898/port_model/'
    
    torch.save(resnet50.state_dict(),save_path)
    # statedic_path=os.path.join(base_path,f'port_model/epoch={epoch}_state.pth')
    # torch.save(resnet50.state_dict(),statedic_path)
if __name__=='__main__':
    # main()
    port_ori()
