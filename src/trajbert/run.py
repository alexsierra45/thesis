import argparse
import torch
from exp.exp_main import Exp_Main as Exp
import time

def main(parser = None):
    if parser == None:
        parser = argparse.ArgumentParser()
    
        # data loader
        parser.add_argument('--root_path', type=str, default='', help='root path') 
        parser.add_argument('--data_path', type=str, default='./data/', help='data path ') 
        
        parser.add_argument('--pre_len', type=str, default='5', help='predict len') 
        parser.add_argument('--data_type', type=str, default='cdr', help='database name')
        parser.add_argument('--infer_data_path', type=str, default='', help='infer data path ') 
        parser.add_argument('--infer_model_path', type=str, default='', help='infer model path ')

        # model define
        parser.add_argument('--d_model', default=512, type=int, help='embed size')
        parser.add_argument('--model',type=str, default='trajbert', help='trajbert')
        parser.add_argument('--head', default=2, type=int, help='multi head num')
        parser.add_argument('--layer', default=2, type=int, help='layer')
        parser.add_argument('--seq_len', default=50, type=int, help='sequence lenght')
        parser.add_argument('--if_posiemb', default=1, type=int, help='position embedding')
        parser.add_argument('--use_his', default=1, type=int,vhelp='use temporal reference') 
        parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
        parser.add_argument('--lradj', type=str, default='type0', help='adjust learning rate')

        # train settings
        parser.add_argument('--is_training', type=int, default=1, help='model is training')
        parser.add_argument('--itr', type=int, default=1, help='experiments times')
        parser.add_argument('--bs', default=256, type=int, help='batch size')
        parser.add_argument('--epoch', default=50, type=int, help='epoch size')
        parser.add_argument('--loss', default='spatial_loss', type=str, help='loss function')
        parser.add_argument('--load_checkpoint', default=0, type=int, help='if continue train')

        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=1, help='gpu')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('Args in experiment:')
    print(args)

    if args.is_training == 1:
        for ii in range(args.itr):
            setting = 'data_{}_{}_dmodel_{}_head_{}_layer_{}_loss_{}_bs_{}_epoch_{}_posiemb_{}_use_temporal_{}_lr_{}'.format(
                args.data_type,
                args.pre_len,
                args.d_model,
                args.head,
                args.layer,
                args.loss,
                args.bs,
                args.epoch,
                args.if_posiemb,
                args.use_his,
                str(args.lr).split('.')[1],
                ii)

            exp = Exp(args) 
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            acc = exp.train(setting)

            torch.cuda.empty_cache()
        return acc
    else:
        for ii in range(args.itr):
            setting = 'data_{}_{}_dmodel_{}_head_{}_layer_{}_loss_{}_bs_{}_epoch_{}_posiemb_{}_use_temporal_{}_lr_{}'.format(
                args.data_type,
                args.pre_len,
                args.d_model,
                args.head,
                args.layer,
                args.loss,
                args.bs,
                args.epoch,
                args.if_posiemb,
                args.relative_v,
                args.remask,
                args.use_his,
                str(args.lr).split('.')[1],
                ii)

            exp = Exp(args)
            print('>>>>>>>start infer : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.infer(setting)

            torch.cuda.empty_cache()
    
if __name__ == "__main__":
    st = time.time()
    main()
    print('spent ', round(time.time() - st, 4), 's')