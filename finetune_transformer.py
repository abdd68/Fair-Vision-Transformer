import torch
import logging
from dataset import *
from model import *
import warnings
from transform import *
from trainAndTest import *
from create_file_list import create_file_list
import argparse

warnings.filterwarnings("ignore")

def count_parameters(model):
    params = sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and 'blocks' in n)
    return params

def create_directory():
	if(not os.path.exists("file_list")):
		os.makedirs("file_list")
	if(not os.path.exists("models")):
		os.makedirs("models")
	if(not os.path.exists("logs")):
		os.makedirs("logs")

def main():
    
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--feature', default='test', help='Remember to change it for parallel tasks.')
    parser.add_argument('--gpu_en', default='0', help='Enable which gpu we use.')

    parser.add_argument('-e','--epochs', type = int, default = 15, help = 'Training epoch.')
    parser.add_argument('-r','--rounds', type = int, default = 2, help = 'In one epoch, we train by rounds.')

    parser.add_argument('-m','--model_type', default='deit_base_patch16_224', help='Assign the model type.')
    parser.add_argument('-d','--dataset_type', default='celeba', help='Assign the dataset type in imagenet,tiny_cifar10,cifar10,cifar100,GTSRB.')
    parser.add_argument('-bs','--batch_size', type = int, default = 32, help='Assign the batch size.')
    parser.add_argument('-ct','--continue_train', action = 'store_true', help = 'Whether to continue training.')
    
    parser.add_argument('-lr','--lr', type = float, default = 0.0005 ,help = 'Learning rate.')
    parser.add_argument('-alr', '--auto_lr',type = float,default=1, help = 'Learning rate in auto mode.')
    parser.add_argument('-mo' ,'--momentum', type = float, default = 0.9, help = 'Momentum.')
    parser.add_argument('-dbg','--debug', action = 'store_true', help = 'Debug mode.')

    parser.add_argument("--classifiers", type=int, nargs='+', default=[8,9,10,11,12], help="The classifiers.") # add
    parser.add_argument("-a", "--alpha", default=0.1, type=float, help="alpha to calculate the dist loss.")
    parser.add_argument("-b", "--bias", default=0, type=float, help="bias to calculate the loss.")

    parser.add_argument("-v","--varsigma",default = [2,2,2,2,2,2,2,2,2,2], nargs='+', type=float, help="consider how much green ratio to add towards the mask.")
    parser.add_argument('-man','--manual', action = 'store_true', help = 'Manual mode, means we do not auto-optimize the magnification value.')
    parser.add_argument('-s','--seed', default = 46, type = int, help = 'Random seed')
    parser.add_argument('--no_norm', action = 'store_true', help = 'Add normalization.')
    parser.add_argument('-g','--gamma', default = 0.5, type = float, help = 'gamma')

    parser.add_argument('-ta','--true_attribute', default = 'Attractive', help = 'The attribute we use to classify.')
    parser.add_argument('-sa','--sensitive_attribute', default = 'Male', help = 'The sensitive attribute.')

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_en
    create_directory()

    logging.basicConfig(level=logging.INFO,  
                    filename='logs/' + args.feature +'_'+ time.strftime('%m-%d-%H:%M:%S',time.localtime(time.time())) + '.log',
                    filemode='w',  
                    format='%(asctime)s: %(message)s'
                    )
    logger.info(args)

    args.nb_classes = 2
    args.nb_s_groups = 2

    create_file_list(args)

    if(args.model_type == 'deit_base_patch16_224'):
         args.num_heads = 12
    else:
        raise('model type error')

    generator = torch.Generator().manual_seed(args.seed)

    dataset_train = SplitDataset("file_list/"+args.feature+"train_filelist.txt", transform_image, 0, 0.9, 'train')
    dataset_val = SplitDataset("file_list/"+args.feature+"train_filelist.txt", transform_image, 0.9, 1, 'val')
    dataset_test = TestDataset("file_list/"+args.feature+"test_filelist.txt", transform_image)
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8 , generator = generator) 
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=8)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=8)
    init_lr = args.lr
    weight_bias = None
    best_acc = 0
    best_DP, best_EO, best_BA = 1,1,0
    for epoch in range(args.epochs):
        args.lr = adjust_learning_rate(init_lr, epoch)
        print(f"lr:{args.lr}")
        train(args, dataloader_train, epoch, weight_bias)

        logger.info(f"train_Acc:{allData_train_test(args, [],[],[], dataloader_train)}")
        output_list, c_list, label_list = [],[],[]
        logger.info(f"val_Acc:{allData_test(args, output_list, c_list, label_list, dataloader_val)}")
        weight_bias = show_2d(args, output_list, c_list, label_list, f'train_val_{epoch}')
        acc = allData_test(args, [], [], [], dataloader_test)
        logger.info(f"test_Acc:{acc}")
        best_acc = max(best_acc,acc)
        if acc > 0.7:
            DP, EO, BA = imbData_test(args, epoch, dataloader_test)
            best_DP, best_EO, best_BA = min(DP, best_DP),min(EO, best_EO),max(BA, best_BA)
        
    logger.info(f"best_DP:{best_DP:.4f}, best_EO:{best_EO:.4f}, best_BA:{best_BA:.4f}")
    logger.info(f"best_acc:{best_acc}")
    
    return
    
if __name__ == '__main__':
	main()