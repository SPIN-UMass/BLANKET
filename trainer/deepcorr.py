from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import warnings
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
# from torchvision import datasets, transforms
import tqdm
import pickle
import pathlib

class PacketWithSizeFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, noise,sizes, inp  , num):
        num = int(num)
        if num ==0:
            return inp
        
        tops = torch.argsort(noise,descending=False)
        
        perts = generate_perturbation(tops[:num])
        
        
        output = inp
        #print (perts,noise.shape,tops.shape)
        adv = torch.ones_like(noise[tops[:num]])*0.001
        output[:,:,0,:] = output[:,:,0,perts]
        output[:,:,4,:] = output[:,:,4,perts]
        output[:,:,0,tops[:num]] = adv
        output[:,:,4,tops[:num]] = 0.595*((sizes[tops[:num]]>0).float()+1)
        
    
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        
        
        return grad_output[:,0,0,:].sum(dim=0),grad_output[:,0,4,:].sum(dim=0), grad_output , None
    
    

def generate_data(dataset,train_index,test_index,flow_size):
    


    global negetive_samples



    all_samples=len(train_index)
    labels=np.zeros((all_samples*(negetive_samples+1),1))
    l2s=np.zeros((all_samples*(negetive_samples+1),1,8,flow_size))

    index=0
    random_ordering=[]+train_index
    for i in tqdm.tqdm( train_index):
        #[]#list(lsh.find_k_nearest_neighbors((Y_train[i]/ np.linalg.norm(Y_train[i])).astype(np.float64),(50)))

        l2s[index,0,0,:]=np.array(dataset[i]['here'][0]['<-'][:flow_size])*1000.0
        l2s[index,0,1,:]=np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0
        l2s[index,0,2,:]=np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0
        l2s[index,0,3,:]=np.array(dataset[i]['here'][0]['->'][:flow_size])*1000.0

        l2s[index,0,4,:]=np.array(dataset[i]['here'][1]['<-'][:flow_size])/1000.0
        l2s[index,0,5,:]=np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0
        l2s[index,0,6,:]=np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0
        l2s[index,0,7,:]=np.array(dataset[i]['here'][1]['->'][:flow_size])/1000.0


        if index % (negetive_samples+1) !=0:
            print (index , len(nears))
            raise
        labels[index,0]=1
        m=0
        index+=1
        np.random.shuffle(random_ordering)
        for idx in random_ordering:
            if idx==i or m>(negetive_samples-1):
                continue

            m+=1

            l2s[index,0,0,:]=np.array(dataset[idx]['here'][0]['<-'][:flow_size])*1000.0
            l2s[index,0,1,:]=np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0
            l2s[index,0,2,:]=np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0
            l2s[index,0,3,:]=np.array(dataset[idx]['here'][0]['->'][:flow_size])*1000.0

            l2s[index,0,4,:]=np.array(dataset[idx]['here'][1]['<-'][:flow_size])/1000.0
            l2s[index,0,5,:]=np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0
            l2s[index,0,6,:]=np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0
            l2s[index,0,7,:]=np.array(dataset[idx]['here'][1]['->'][:flow_size])/1000.0

            #l2s[index,0,:,0]=Y_train[i]#np.concatenate((Y_train[i],X_train[idx]))#(Y_train[i]*X_train[idx])/(np.linalg.norm(Y_train[i])*np.linalg.norm(X_train[idx]))
            #l2s[index,1,:,0]=X_train[idx]



            labels[index,0]=0
            index+=1
    return l2s, labels

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2000, (2,20), stride=2)
        self.max_pool1 = nn.MaxPool2d((1,5), stride=1)
        
        self.conv2 = nn.Conv2d(2000, 800, (4,10), stride=2)
        self.max_pool2 = nn.MaxPool2d((1,3), stride=1)
        
        self.fc1 = nn.Linear(49600, 3000)
        self.fc2 = nn.Linear(3000, 800)
        self.fc3 = nn.Linear(800, 100)
        self.fc4 = nn.Linear(100, 1)
#         self.d = nn.Dropout2d()
    
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, inp, dropout_prob):
        x = inp
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        
        x = x.view(batch_size, -1)
        
        
        x = F.relu(self.fc1(x))
        x = F.dropout2d(x, p=dropout_prob)
        x = F.relu(self.fc2(x))
        x = F.dropout2d(x, p=dropout_prob)
        x = F.relu(self.fc3(x))
        x = F.dropout2d(x, p=dropout_prob)
        x = self.fc4(x)
        return x
        
        
class TIMENOISER(nn.Module):
    def __init__(self,inp):
        super(TIMENOISER, self).__init__()

        self.inp = inp

        self.independent = nn.Sequential(
            nn.Linear(inp,500),
            nn.ReLU(),
            nn.Linear(500,inp)
        )
#         self.d = nn.Dropout2d()
    
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, inp,eps,mid,outsize = 300):
        x = inp[:,:self.inp]
        nz = torch.ones_like(x)
        nz.uniform_(-2,2)
        ind = self.independent(nz)
    
        res = ind
        res = transfer_adv(res,eps,mid)
        
        res = res.view(-1,1,self.inp)
        
        
        
        if self.inp < outsize :
            z = torch.zeros_like(res)
            res = torch.cat([res,z,z,z,z,z],dim=2)
            res = res[:,:,:outsize]
            
        
        z=torch.zeros_like(res)
        x= torch.stack([res,z,z,z,z,z,z,z],dim=2)
        
        
        return x
class ADDNOISER(nn.Module):
    def __init__(self,inp,device):
        super(ADDNOISER, self).__init__()
        
        self.inp = inp
        
        self.z = torch.FloatTensor(size=(1,inp))
        self.z =self.z.to(device)
        self.independent_where = nn.Sequential(
            nn.Linear(inp,500),
            nn.ReLU(),
            nn.Linear(500,inp)
        )
        self.independent_size = nn.Sequential(
            nn.Linear(inp,500),
            nn.ReLU(),
            nn.Linear(500,inp)
        )
#         self.d = nn.Dropout2d()
    
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()
    
    def forward(self,outsize=300):
        
        nz = self.z
        nz.uniform_(-0,0.5)
        #print (nz.shape)
        ind_where = self.independent_where(nz)
        ind_size = self.independent_size(nz)
        
        if self.inp < outsize :
            z = torch.zeros_like(ind_where)
            ind_size = torch.cat([ind_size,z,z,z,z,z],dim=1)
            ind_where = torch.cat([ind_where,z,z,z,z,z],dim=1)
        ind_size = ind_size[:,:outsize]
        ind_where = ind_where[:,:outsize]
        
        
        
        return ind_where.view(-1),ind_size.view(-1)
        
        
class discrim(nn.Module):
    def __init__(self,inp):
        super(discrim, self).__init__()
        self.inp = inp
        self.dependent = nn.Sequential(
                nn.Linear(inp,1000),
                nn.ReLU(),
                nn.Linear(1000,1000),
                nn.ReLU(),
                nn.Linear(1000,1)
            )
    
    
    def forward(self, inp):
        return  self.dependent(inp[:,0,0,:self.inp])

def generate_perturbation(change_points, size = 300):
    start = size-len(change_points)
    pert = [] 
    passed = 0 
    for ind in range(size):
        if ind in change_points:
            pert.append(start)
            start +=1
            passed+=1
        else:
            pert.append(ind-passed)
        
    return pert



decider= PacketWithSizeFunction.apply
def train_adv(adv_model,size_model,optim_adv,disc_model,opt_dis, model, device, data, label,num_to_add= 0,reg = 0 ,eps=0.3,mid=5.0,deps = 300):
    model.eval()
    model.zero_grad()
    
    #data[:,:,0,:] = torch.clamp(data[:,:,0,:] + transfer_adv(adv_model,eps,mid), min = 0.0)
    
    #print (adv.shape)
    if reg > 0:
        where,sizes = size_model()
    

        data_adv = decider(where,sizes,data,num_to_add)

        adv  = adv_model(data_adv[:,0,0,:deps],eps,mid)
        z = torch.zeros_like(adv )
        z.normal_(mean=mid,std=eps)
        fake = disc_model(adv)
        real = disc_model(z)
        #print ( real.shape,fake.shape,label.shape)
    #     
        f_loss = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(label))
        r_loss = F.binary_cross_entropy_with_logits(real, torch.ones_like(label))

        d_loss = f_loss + r_loss
        opt_dis.zero_grad()
        d_loss.backward()
        opt_dis.step()

    
    where,sizes = size_model()
        
    data_adv = decider(where,sizes,data,num_to_add)

    adv  = adv_model(data_adv[:,0,0,:deps],eps,mid)
    output = model(torch.clamp(data_adv+adv,min=0), 0.0)
    #output_correct = model(data, 0.4)
    fake = disc_model(adv)
    loss = (F.binary_cross_entropy_with_logits(output,1 - label)) + reg*F.binary_cross_entropy_with_logits(fake, torch.ones_like(label)) 
    
    optim_adv.zero_grad()
    loss.backward()
    optim_adv.step()
    
    
def gen_advs(adv_model,optim_adv,disc_model,opt_dis, model, device, data, label, lr, alpha=0.5, eps=0.3,mid=5.0,deps = 300):
    model.eval()
    model.zero_grad()
    
    #data[:,:,0,:] = torch.clamp(data[:,:,0,:] + transfer_adv(adv_model,eps,mid), min = 0.0)
    adv  = adv_model(data[:,0,0,:deps],eps,mid)
    z = torch.zeros_like(adv )
    z.normal_(mean=mid,std=eps)
    return adv,z
    
    

    
#     print ('Mean:  ',transfer_adv(adv_model, eps, mid).mean())
#     print ('STD:   ',transfer_adv(adv_model, eps, mid).std())


def test_adv(adv_model,size_model, model, device, data, num_to_add,eps,mid,deps):
    model.eval()
    model.zero_grad()
#     time = data[:,:,0:4,:]
#     size = data[:,:,4:8,:]
#     time = time + eps*(torch.sigmoid(adv_model)-0.5)
#     adv_data = torch.cat((time, size), dim = 2)
    where,sizes = size_model()
        
    data_adv = decider(where,sizes,data,num_to_add)

    adv  = adv_model(data_adv[:,0,0,:deps],eps,mid)
    
    output_adv = model(torch.clamp(data_adv+adv,min=0), 0.0)
    
    o = torch.sigmoid(output_adv)
    return o
    
    
def total_test(adv_model,size_model, model, device, num_to_add,eps,mid,deps):
    global test_index
    global dataset
    global batch_size
    a = -1
    corrs=np.zeros((500,500))
    batch=[]
    l2s_test_all=np.zeros((batch_size,1,8,flow_size))
    l_ids=[]
    index=0
    xi,xj=0,0
    for i in (test_index[:500]):
        xj=0
        for j in test_index[:500]:

            l2s_test_all[index,0,0,:]=np.array(dataset[j]['here'][0]['<-'][:flow_size])*1000.0
            l2s_test_all[index,0,1,:]=np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0
            l2s_test_all[index,0,2,:]=np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0
            l2s_test_all[index,0,3,:]=np.array(dataset[j]['here'][0]['->'][:flow_size])*1000.0

            l2s_test_all[index,0,4,:]=np.array(dataset[j]['here'][1]['<-'][:flow_size])/1000.0
            l2s_test_all[index,0,5,:]=np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0
            l2s_test_all[index,0,6,:]=np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0
            l2s_test_all[index,0,7,:]=np.array(dataset[j]['here'][1]['->'][:flow_size])/1000.0
            l_ids.append((xi,xj))
            index+=1
            if index==batch_size:
                index=0
                test_data = torch.from_numpy(l2s_test_all).float().to(device)
                cor_vals=test_adv(adv_model,size_model, model, device, test_data,num_to_add, eps, mid,deps)
#                 cor_vals = test(model, device, test_data)
                cor_vals = cor_vals.data.cpu().numpy()
                for ids in range(len(l_ids)):
                    di,dj=l_ids[ids]
                    corrs[di,dj]=cor_vals[ids]
                l_ids=[]
            xj+=1
        xi+=1
    return (corrs)

negetive_samples=199
flow_size=300
TRAINING= True


all_runs={'8872':'192.168.122.117',
           '8802':'192.168.122.117','8873':'192.168.122.67','8803':'192.168.122.67',
          '8874':'192.168.122.113','8804':'192.168.122.113','8875':'192.168.122.120',
         '8876':'192.168.122.30','8877':'192.168.122.208','8878':'192.168.122.58'}



dataset=[]

for name in all_runs:
    dataset+=pickle.load(open('/home/abahramali/deepsec/fgsm/deepcorr-data/%s_tordata300.pickle'%name, 'rb'))
    
if TRAINING:
    
    
    len_tr=len(dataset)
    train_ratio=float(len_tr-3000)/float(len_tr)
    rr= list(range(len(dataset)))
    np.random.shuffle(rr)

    train_index=rr[:int(len_tr*train_ratio)]
    test_index= rr[int(len_tr*train_ratio):int(len_tr*train_ratio)+500] #range(len(dataset_test)) # #
    pickle.dump(test_index,open('test_index300.pickle','wb'))
    pickle.dump(train_index, open('train_index300.pickle', 'wb'))
else:
    test_index=pickle.load(open('test_index300.pickle'))
    
    
    
    


parser = argparse.ArgumentParser(description='DEEPCORR BLIND ADV EXAMPLE')

parser.add_argument('--gpu-id', type=int, default=0, help='Train model')
parser.add_argument('--mid', type=float, default=0.0, help='Train model')
parser.add_argument('--sigma', type=float, default=10, help='Train model')
parser.add_argument('--to-add', type=float, default=0, help='Train model')
parser.add_argument('--similarity-reg', type=float, default=0.0, help='Train model')
parser.add_argument('--input-size', type=int, default=300, help='Train model')
parser.add_argument('--epochs', type=int, default=20, help='Train model')

parser.add_argument('--gen-lr', type=float, default=0.001, help='Train model')
parser.add_argument('--dis-lr', type=float, default=0.0001, help='Train model')
parser.add_argument('--justpos', type=int, default=0, help='Train model')




        
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu", 0)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print (device)


model = Net().to(device)
LOADED=torch.load('/home/abahramali/deepsec/general-noise/deepcorr-epoch20-step779-acc0.87.pth.tar', map_location={'cuda:3': 'cuda:0'})
model.load_state_dict(LOADED)



num_epoches = 200
batch_size = 100




l2s,labels=generate_data(dataset=dataset,train_index=train_index,test_index=test_index,flow_size=flow_size)
rr = list(range(len(l2s)))
np.random.shuffle(rr)
l2s = l2s[rr]
labels = labels[rr]

num_steps = (len(l2s)//batch_size)



###### PARAMETERS
args = parser.parse_args()


inpsize = args.input_size
noise_lr = args.gen_lr
disc_lr = args.dis_lr
gpu_id = args.gpu_id
mid = args.mid
sigma = args.sigma
num_to_add  = args.to_add
reg = args.similarity_reg
epochs = args.epochs


def transfer_adv(inp,eps,mid):
#     x = 100*F.sigmoid(inp)
#     return F.relu(inp)
    
    x = inp
    if args.justpos == 1 :
        x = F.relu(x)
    res = ((x-torch.clamp(x.mean(dim=1,keepdim=True)-mid,min=0)-torch.clamp(x.mean(dim=1,keepdim=True)+mid,max=0)))

    res_multi = (torch.clamp(x.std(dim=1,keepdim=True),max=eps)/(x.std(dim=1,keepdim=True))+0.000001)
    res = res * res_multi
    #print (res)
    return  res



SAVE_PATH='outputs_new_deepcorr/mid_%d/sigma_%d/add_%d/reg_%0.5f/inpsize_%d/just_positive_%d/'%(mid,sigma,num_to_add,reg,inpsize,args.justpos)



pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)








timenois = TIMENOISER(inpsize)
timenois.to(device)
addnois = ADDNOISER(inpsize,device)
addnois.to(device)
optim_nos = optim.Adam(list(addnois.parameters())+list(timenois.parameters()),lr=noise_lr)
disc  = discrim(inpsize)
disc.to(device)
optim_dis = optim.Adam(disc.parameters(),lr=disc_lr)
print (num_steps)



total_corrs = []
for epoch in range(epochs):
    rr = list(range(len(l2s)))
    np.random.shuffle(rr)
    l2s = l2s[rr]
    labels = labels[rr]
    print ('EPOCH %d'%epoch)
    for step in range(500):
        
        start_ind = step*batch_size
        end_ind = ((step + 1) *batch_size)
        if end_ind < start_ind:
            print ('HOOY')
            continue
            
        else:
            batch_flow = torch.from_numpy(l2s[start_ind:end_ind, :]).float().to(device)
            batch_label = torch.from_numpy(labels[start_ind:end_ind]).float().to(device)
    
        train_adv(timenois,addnois,optim_nos,disc,optim_dis, model, device, batch_flow, batch_label,num_to_add, reg,sigma,mid,300)
        
    corrs = total_test(timenois,addnois, model, device,num_to_add, sigma,mid,300)
    torch.save({'time_model':timenois.state_dict(),'add_model':addnois.state_dict()},SAVE_PATH+'/model_epoch_%d'%epoch)
    
    total_corrs.append(corrs)
    
    torch.save({'corrs':total_corrs},SAVE_PATH+'/cors')










