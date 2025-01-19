import os
import pickle
import numpy as np
import torch
import tqdm
import torch.nn as nn
from torch import optim

from data_factory import data_provider
from exp.exp_basic import Exp_Basic
from loss import Loss_Function
from model.bert import TrajBERT
from utils import adjust_learning_rate, get_evalution

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.data_provider = data_provider(args)
        self.vocab_size = self.data_provider.get_vocabsize()
        self.exchange_map = self.data_provider.get_coocurrence_map()
        self.model: nn.Module = self._build_model(self.vocab_size).to(self.device)

    def _build_model(self,vocab_size):
        model_dict = {
            'trajbert': TrajBERT
        }
        model = model_dict[self.args.model](args = self.args,vocab_size = vocab_size).float()

        return model
    
    def _select_optimizer(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def _select_criterion(self):
        if self.args.loss == 'loss':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = Loss_Function()
        return criterion
    
    def train(self, setting):
        train_loader = self.data_provider.get_loader(flag='train',args = self.args)

        test_loader = self.data_provider.get_loader(flag='test',args = self.args)

        if self.args.load_checkpoint and os.path.exists(self.args.root_path + '/checkpoint/' + setting + '.pth'):
            state_dict = torch.load(self.args.root_path + '/checkpoint/' + setting + '.pth', map_location=self.device)
            self.model.load_state_dict(state_dict['model'])
            print('load model ' + self.args.root_path + '/checkpoint/' + setting + '.pth' + ' success')
        
        print('start train')
        
        best_score, score_data = self.get_best_score(setting)
        tmp_best_score = 0

        model_optim = self._select_optimizer() 
        criterion = self._select_criterion()

        for epoch in range(self.args.epoch):
            train_loss = []
            self.model.train()

            train_loader = self.data_provider.get_loader(flag='train', args = self.args)
            
            for _, (input_ids, masked_tokens, masked_pos, user_ids, 
                    day_ids, input_next, input_prior, 
                    input_prior_dis, input_next_dis, input_mask) in enumerate(tqdm.tqdm(train_loader, ncols=100)):
                model_optim.zero_grad()
                logits_lm = self.model(input_ids, masked_pos, user_ids, day_ids, 
                                       input_next, input_prior, input_prior_dis, 
                                       input_next_dis, input_mask)
               
                loss = self.calculate_loss(logits_lm, masked_tokens, criterion)

                train_loss.append(loss)
                
                loss.backward()
                model_optim.step()

            train_loss = torch.mean(torch.stack(train_loss))
            print("Epoch: {} | Train Loss: {}  ".format(epoch + 1, train_loss))
            if (epoch + 1) % 2 == 0:
                
                torch.save({'model': self.model.state_dict()}, self.args.root_path + '/checkpoint/' + setting + '.pth')
                result, test_loss, accuracy_score, wrong_pre, predictions = self.test(test_loader, criterion)
                tmp_best_score = max(tmp_best_score, accuracy_score)
                if accuracy_score >= best_score:
                    torch.save({'model': self.model.state_dict()}, self.args.root_path + '/result/' + setting + '.pth')
                    print(f'update best score from {best_score} to {accuracy_score}')
                    best_score = accuracy_score
                    score_data[setting] = best_score
                    pickle.dump(score_data, open(self.args.root_path + '/best_score/best_score.pkl', 'wb+'))

                f = open(self.args.root_path + '/result/' + setting + '.txt', 'a+')
                f.write("epoch: %d \n" % (epoch + 1))
                f.write("train loss: %.6f | test loss: %.6f \n" % (train_loss, test_loss))
                f.write(result)
                f.close()
                print("Epoch: {} | Train Loss: {}  Test Loss: {}".format(epoch + 1, train_loss, test_loss))

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        print('best accuracy:', tmp_best_score)
        return tmp_best_score
    
    def test(self, test_loader, criterion):
        total_loss = []
        self.model.eval()
        predict_prob = None  # Inicialización en None
        total_masked_tokens = []  # Cambiar a lista para evitar acumulación innecesaria en memoria GPU
        predictions = []

        with torch.no_grad():
            for i, (input_ids, masked_tokens, masked_pos, user_ids, 
                    day_ids, input_next, input_prior, 
                    input_prior_dis, input_next_dis, input_mask) in enumerate(tqdm.tqdm(test_loader, ncols=100)):

                # Pasa los datos a través del modelo
                logits_lm = self.model(input_ids, masked_pos, user_ids, day_ids, 
                                    input_next, input_prior, input_prior_dis, 
                                    input_next_dis, input_mask)
                
                # Ajustamos la máscara `input_mask` para que coincida con `logits_lm`
                batch_size, seq_len, num_classes = logits_lm.size()
                valid_mask = input_mask[:, :seq_len].unsqueeze(2).expand(-1, -1, num_classes).bool()

                # Filtra logits_lm y masked_tokens utilizando valid_mask
                logits_lm_filtered = logits_lm[valid_mask]
                masked_tokens_filtered = masked_tokens[input_mask.bool()]  # Filtrar tokens válidos

                # Convertir a CPU y añadir a total_masked_tokens
                masked_tokens_filtered_cpu = masked_tokens_filtered.cpu().numpy()
                total_masked_tokens.extend(masked_tokens_filtered_cpu.flatten().tolist())

                # Reorganizar logits_lm_filtered si es unidimensional
                if logits_lm_filtered.dim() == 1:
                    logits_lm_filtered = logits_lm_filtered.view(-1, num_classes)

                logits_lm_ = torch.topk(logits_lm_filtered, 100, dim=1)[1]
                logits_lm_topk = torch.topk(logits_lm_filtered, 100, dim=1)
                logits_lm_values = logits_lm_topk[0].cpu()
                logits_lm_indices = logits_lm_topk[1].cpu()

                # Guardamos las predicciones del batch en la CPU
                batch_predictions = {
                    "input_ids": input_ids.cpu().numpy(),
                    "masked_tokens": masked_tokens_filtered_cpu,
                    "predicted_indices": logits_lm_indices.numpy(),
                    "predicted_probs": logits_lm_values.numpy()
                }
                predictions.append(batch_predictions)

                # Concatenar predict_prob en la CPU para evitar saturar la GPU
                if predict_prob is None:
                    predict_prob = logits_lm_.cpu()  # Mover a CPU
                else:
                    predict_prob = torch.cat([predict_prob, logits_lm_.cpu()], dim=0)

                # Calcula la pérdida solo en las posiciones válidas
                loss = self.calculate_loss(logits_lm_filtered, masked_tokens_filtered, criterion)

                total_loss.append(loss)  # Asegúrate de mover el loss a la CPU

                # Liberar memoria GPU
                del logits_lm, logits_lm_filtered, valid_mask
                torch.cuda.empty_cache()

        self.model.train()
        total_loss = torch.mean(torch.stack(total_loss))

        accuracy_score, fuzzzy_score, top3_score, top5_score, top10_score, top30_score, top50_score, top100_score, map_score, wrong_pre = get_evalution(
        ground_truth=total_masked_tokens, logits_lm=predict_prob, exchange_matrix=self.exchange_map)

        return 'test accuracy score = ' + '{:.6f}'.format(accuracy_score) + '\n' \
            + 'fuzzzy score = ' +  '{:.6f}'.format(fuzzzy_score) + '\n'\
            + 'test top3 score = '+ '{:.6f}'.format(top3_score) + '\n'\
            + 'test top5 score = '+ '{:.6f}'.format(top5_score) + '\n'\
            + 'test top10 score = '+ '{:.6f}'.format(top10_score) + '\n'\
            + 'test top30 score = '+ '{:.6f}'.format(top30_score) + '\n'\
            + 'test top50 score = '+ '{:.6f}'.format(top50_score) + '\n'\
            + 'test top100 score = '+ '{:.6f}'.format(top100_score) + '\n' \
            + 'test MAP score = '+ '{:.6f}'.format(map_score) + '\n' , total_loss, accuracy_score, wrong_pre, predictions
        
    def infer(self, setting):
        infer_result_path = self.args.root_path + '/infer_result'
        if not os.path.exists(infer_result_path):
            os.mkdir(infer_result_path)

        if not os.path.exists(self.args.root_path + '/result/' + setting + '.pth'):
            print('no such model, check args settings')
            return

        if self.args.infer_model_path == '':
            self.load_weight(self.args.root_path + '/result/' + setting + '.pth')
            print('load model ' + self.args.root_path + '/result/' + setting+'.pth' + ' success')

        else:
            self.load_weight(self.args.root_path + '/result/' + self.args.infer_model_path)
            print('load model ' + self.args.root_path + '/result/' + self.args.infer_model_path + ' success')
        
        test_loader = self.data_provider.get_loader(flag='infer', args = self.args)
        criterion = self._select_criterion()
        result, test_loss, accuracy_score, wrong_pre, predictions = self.test(test_loader, criterion)

        f = open(self.args.root_path + '/infer_result/' + setting + '.txt', 'a+')
        f.write("test loss: %.6f \n" %  test_loss)
        f.write(result)
        f.write('\n'.join(wrong_pre))
        f.close()

        return
    
    def calculate_loss(self, logits_lm, masked_tokens, criterion):
        if self.args.loss == "spatial_loss":
            loss_lm = criterion.Spatial_Loss(self.exchange_map, logits_lm.view(-1, self.vocab_size),
                                            masked_tokens.view(-1))
        else:
            loss_lm = criterion(logits_lm.view(-1, self.vocab_size), masked_tokens.view(-1))
        loss = (loss_lm.float()).mean()

        return loss
    
    def get_best_score(self,setting):
        middata_path = self.args.root_path + '/best_score'
        best_model_path = self.args.root_path + '/result'
        checkpoint_path = self.args.root_path + '/checkpoint'
        if not os.path.exists(middata_path):
            os.mkdir(middata_path)
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        if not os.path.exists(best_model_path):
            os.mkdir(best_model_path)

        best_score_file = middata_path + '/best_score.pkl'
        if not os.path.exists(best_score_file):
            score_data = {setting:0}
            pickle.dump(score_data,open(best_score_file, 'wb+'))
        score_data = pickle.load(open(best_score_file, 'rb'))
        if setting not in score_data:
            score_data[setting] = 0
        best_score = score_data[setting]
        print(setting, ' history best score ', best_score)

        return best_score, score_data
    
    def load_weight(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)['model']
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)