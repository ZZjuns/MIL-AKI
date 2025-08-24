"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import copy
import os
import random
from typing import Callable, Dict, Tuple, Optional, Union, List, cast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score, f1_score, recall_score, precision_recall_curve, auc,
)
from timm.scheduler.scheduler import Scheduler
from torch import nn
from torch.nn.functional import sigmoid
from torch.optim import Optimizer
from tqdm import tqdm

from millet.data.mil_tsc_dataset import MILTSCDataset
from millet.interpretability_metrics import calculate_aopcr, calculate_ndcg_at_n
from millet.util import *
import matplotlib.pyplot as plt
from timm.optim.adamp import AdamP
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MILLETModel:
    """Wrapper for models in the MILLET framework."""

    def __init__(self, name: str, device: torch.device, n_classes: int, net: nn.Module):
        super().__init__()
        self.name = name
        self.device = device
        self.n_classes = n_classes
        self.net = net.to(self.device)

    def fit(
        self,
        train_dataset: MILTSCDataset,
        test_dataset: MILTSCDataset,
        args,
        n_epochs: int,
        batch_size: int = 16,
        dropout_patch: int = 0,
        criterion: Callable = cross_entropy_criterion,
        optimizer: Optimizer = None,
    ) -> None:
        """
        Fit the MILLET model.

        :param train_dataset: MIL TSC dataset to fit to.
        :param n_epochs: Number of epochs to train for.
        :param learning_rate: Learning rate of optimizer.
        :param weight_decay: Weight decay of optimizer.
        :param criterion: Loss function to train to minimise.
        :return:
        """
        print(self.device)
        # args.save_dir = args.save_dir + 'InceptBackbone'
        save_dir = args.save_dir + args.model_name
        # args.save_dir = args.save_dir + 'ResNetBackbone'

        maybe_mkdir_p(join(save_dir, f'{args.dataset}'))
        save_dir = make_dirs(join(save_dir, f'{args.dataset}'))
        maybe_mkdir_p(save_dir)

        # <------------- set up logging ------------->
        logging_path = os.path.join(save_dir, 'Train_log.log')
        # 移除所有的 handler
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logger = get_logger(logging_path)

        # <------------- save hyperparams ------------->
        option = vars(args)
        file_name = os.path.join(save_dir, 'option.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(option.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        # Setup
        torch_train_dataloader = train_dataset.create_dataloader(shuffle=True, batch_size=batch_size)

        # ReduceLROnPlateau 调度器
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.01, verbose=True)

        # Early stopping setup
        best_net = None
        # best_loss = np.Inf
        best_roc = 0
        best_score = 0
        best_dict = {}
        num_patience = 0
        # 用于记录每个 epoch 的训练损失和测试损失
        train_losses = []
        test_losses = []

        save_path = join(save_dir, 'weights')
        os.makedirs(save_path, exist_ok=True)

        # Train over multiple epochs
        for index in range(n_epochs):
            self.net.train()
            total_loss = 0
            with tqdm(total=len(torch_train_dataloader), desc="batch", dynamic_ncols=True, leave=False) as batch_bar:
                # Train model for an epoch
                for batch in torch_train_dataloader:
                    bags = batch["bags"]
                    targets = batch["targets"].to(self.device)
                    if isinstance(bags, list):
                        bags = torch.stack(bags)
                    bags = bags.to(self.device)

                    # window-based random masking
                    if dropout_patch > 0:
                        selecy_window_indx = random.sample(range(10), int(dropout_patch * 10))
                        inteval = int(len(bags) // 10)
                        for idx in selecy_window_indx:
                            bags[:, idx * inteval:idx * inteval + inteval, :] = torch.randn(1).cuda()

                    optimizer.zero_grad()
                    model_out= self(bags)
                    loss = criterion(model_out["bag_logits"], targets)
                    loss.backward()
                    total_loss += loss

                    # avoid the overfitting by using gradient clip
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2.0)

                    optimizer.step()
                    batch_bar.update(1)

            # Evaluate
            self.net.eval()
            epoch_test_results = self.evaluate(args, test_dataset, criterion)
            logger.info(
                '\r Epoch [%d/%d]  train loss: %.4f  test loss: %.4f  accuracy: %.4f  bal. average score: %.4f  roc_auc ovo marco: %.4f  roc_auc ovr marco: %.4f  f1_marco:%.4f  recall:%.4f  auprc:%.4f' %
                (index+1, args.n_epochs, total_loss/len(torch_train_dataloader), epoch_test_results['loss'], epoch_test_results['acc'], epoch_test_results['bal_acc'], epoch_test_results['roc_auc_ovo_marco'],epoch_test_results['roc_auc_ovr_marco'],
                 epoch_test_results['f1_marco'],epoch_test_results['r_marco'],epoch_test_results['auprc']))


            # 计算并存储每个 epoch 的平均训练损失
            avg_train_loss = total_loss / len(torch_train_dataloader)
            avg_train_loss = avg_train_loss.detach().cpu().numpy()
            train_losses.append(avg_train_loss)
            avg_test_loss = epoch_test_results['loss']
            test_losses.append(avg_test_loss.cpu())  # 记录测试集损失
            # Handle early stopping
            epoch_roc = epoch_test_results["roc_auc_ovo_marco"]
            if epoch_roc > best_roc:
                best_net = copy.deepcopy(self.net)
                best_roc = epoch_roc
                best_dict = epoch_test_results
                num_patience = 0
                print("best_roc:{},best_acc:{}".format(best_roc,best_dict['acc']))

                # 保存结果为 CSV 文件
                best_dict = pd.DataFrame([best_dict])
                csv_file_path = os.path.join(save_dir, 'best_dict.csv')
                # 检查文件是否存在
                if not os.path.isfile(csv_file_path):
                    # 如果文件不存在，则保存为新的 CSV 文件（包含 header）
                    best_dict.to_csv(csv_file_path, index=False)
                else:
                    # 如果文件存在，则追加数据（不写入 header）
                    best_dict.to_csv(csv_file_path, mode='a', index=False, header=False)
                print("save best dict to" + csv_file_path)

                #    保存模型
                save_name = os.path.join(save_path, 'best_model.pth')
                self.save_weights(save_name)
            else:
                num_patience += 1

            # if num_patience >= patience:
            #     print("早停止")
            #     break

        # 绘制损失曲线图
        plt.plot(range(index+1), train_losses, label='Training Loss')
        plt.plot(range(index+1), test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Epoch vs Loss')
        plt.legend()
        plt.show()
        return epoch_test_results


    def evaluate(
        self,
        args,
        dataset: MILTSCDataset,
        criterion: Callable = cross_entropy_criterion,
    ) -> Dict:
        # Iterate through data loader and gather preds and targets
        all_bag_logits_list = []
        all_targets_list = []
        # Don't need to worry about batch size being too big during evaluation (only training)
        dataloader = dataset.create_dataloader(batch_size=64, shuffle=True)
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                bags = batch["bags"]
                targets = batch["targets"].to(self.device)
                model_out = self(bags)
                bag_logits = model_out["bag_logits"]

                loss = criterion(bag_logits, targets)
                bag_loss = loss
                total_loss += bag_loss

                all_bag_logits_list.append(torch.sigmoid(bag_logits).cpu().numpy())
                # all_bag_logits_list.append(bag_logits.cpu().numpy())
                all_targets_list.append(targets.cpu().numpy())


        # Gather bag logits and targets into tensors
        all_targets = np.vstack(all_targets_list)
        all_bag_logits = np.vstack(all_bag_logits_list)

        # Get probas from logits
        all_pred_probas = np.exp(all_bag_logits) / np.sum(np.exp(all_bag_logits), axis=1, keepdims=True)

        # If in binary case, reduce probas to single prediction (not doing so breaks some of the evaluation metrics)
        if all_pred_probas.shape[1] == 2:
            all_pred_probas = all_pred_probas[:, 1]

        # Get the actual predicted classes
        # _, all_pred_clzs = torch.max(all_bag_logits, dim=1)
        all_pred_clzs = np.argmax(all_bag_logits, axis=1)
        all_targets = np.argmax(all_targets, axis=1)


        # Compute metrics
        acc = accuracy_score(all_targets, all_pred_clzs)
        bal_acc = balanced_accuracy_score(all_targets, all_pred_clzs)

        roc_auc_ovo_marco = roc_auc_score(all_targets, all_pred_probas, average='macro', multi_class='ovo')
        # roc_auc_ovo_marco = 0
        # roc_auc_ovo_micro = roc_auc_score(all_targets,all_pred_probas,average='micro',multi_class='ovo')
        roc_auc_ovo_micro = 0
        roc_auc_ovr_marco = roc_auc_score(all_targets, all_pred_probas, average='macro', multi_class='ovr')
        # roc_auc_ovr_marco = 0
        # roc_auc_ovr_micro = roc_auc_score(all_targets, all_pred_probas, average='micro', multi_class='ovr')
        roc_auc_ovr_micro = 0
        # conf_mat = torch.as_tensor(confusion_matrix(all_targets, all_pred_probas), dtype=torch.float)

        f1_marco = f1_score(all_targets, all_pred_clzs, average='macro')
        f1_micro = f1_score(all_targets, all_pred_clzs, average='micro')

        r_marco = recall_score(all_targets, all_pred_clzs, average='macro')
        r_micro = recall_score(all_targets, all_pred_clzs, average='micro')

        # 计算AOPRC
        precision, recall, thresholds = precision_recall_curve(all_targets, all_pred_probas)
        AUPRC = auc(recall, precision)

        # Return results in dict
        all_results = {
            "loss": total_loss / len(dataloader),
            "acc": acc,
            "roc_auc_ovo_marco": roc_auc_ovo_marco,
            "roc_auc_ovo_micro": roc_auc_ovo_micro,
            "roc_auc_ovr_marco": roc_auc_ovr_marco,
            "roc_auc_ovr_micro": roc_auc_ovr_micro,
            "f1_marco": f1_marco,
            "f1_micro": f1_micro,
            "r_marco": r_marco,
            "r_micro": r_micro,
            "auprc": AUPRC,
            "bal_acc": bal_acc,
            # "conf_mat": conf_mat,
        }
        return all_results

    def evaluate_interpretability(
        self,
        dataset: MILTSCDataset,
    ) -> Tuple[float, Optional[float]]:
        all_aopcrs = []
        all_ndcgs = []
        # Don't need to worry about batch size being too big during evaluation (only training)
        dataloader = dataset.create_dataloader(batch_size=16,shuffle=True)
        with torch.no_grad():
            for batch in custom_tqdm(dataloader, leave=False):
                bags = batch["bags"]
                batch_targets = batch["targets"]
                # Calculate AOPCR for batch
                batch_aopcr, _, _ = calculate_aopcr(self, bags, verbose=False)
                all_aopcrs.extend(batch_aopcr.tolist())
                # Calculate NDCG@n for batch if instance targets are present
                if "instance_targets" in batch:
                    batch_instance_targets = batch["instance_targets"]
                    all_instance_importance_scores = self.interpret(self(bags))
                    for bag_idx, bag in enumerate(bags):
                        target = batch_targets[bag_idx]
                        instance_targets = batch_instance_targets[bag_idx]
                        ndcg = calculate_ndcg_at_n(
                            all_instance_importance_scores[bag_idx, target],
                            instance_targets,
                        )
                        all_ndcgs.append(ndcg)
        avg_aopcr = np.mean(all_aopcrs)
        avg_ndcg = float(np.mean(all_ndcgs)) if len(all_ndcgs) > 0 else None
        return float(avg_aopcr), avg_ndcg

    def interpret(self, model_out: Dict) -> torch.Tensor:
        return model_out["interpretation"]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def save_weights(self, path: str) -> None:
        # Save net from CPU
        #  Fixes issues with saving and loading from different devices
        print("Saving model to {:s}".format(path))
        torch.save(self.net.to("cpu").state_dict(), path)
        # Ensure net is back on original device
        self.net.to(self.device)

    def load_weights(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()

    def forward(
        self, bag_input: Union[torch.Tensor, List[torch.Tensor]], bag_instance_positions: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Get the model output for a bag input.

        :param bag_input: Either a single bag or a batch of bags.
        :param bag_instance_positions: Single or batch of instance positions for each bag.
        :return: The model output.
        """
        # Reshape input depending on whether we have a single bag or a batch
        bags, is_unbatched_bag = self._reshape_bag_input(bag_input)
        # Actually pass the input through the model
        model_output = self._internal_forward(bags, bag_instance_positions)
        # If given input was not batched, un-batch the output
        if is_unbatched_bag:
            unbatched_model_output = {}
            for key, value in model_output.items():
                unbatched_model_output[key] = value[0]
            return unbatched_model_output
        return model_output

    def _reshape_bag_input(
        self, bag_input: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], bool]:
        """
        Converts bag inputs to a consistent type.

        Input options:
        * 2D tensor (one unbatched bag - dims: n_instance x d_instance)
        * 3D tensor (batched collection of bags - dims: n_batch x n_instance x d_instance)
        * List of 2D tensors (batched collection of bags, but batch dim is the list not a tensor dim).

        The output is a batched collection on bags. For each input:
        * 2D tensor -> [2D Tensor]
        * 3D tensor -> 3D tensor
        * List [2D tensor] -> List [2D tensor]

        Note only the 2D tensor needs to be reshaped as it is unbatched.

        :param bag_input: See input options.
        :return: See outputs for each input above.
        """
        reshaped_input: Union[torch.Tensor, List[torch.Tensor]]
        # Bag input is a tensor
        if torch.is_tensor(bag_input):
            # Enforce that we're now using bag_input as a tensor
            bag_input = cast(torch.Tensor, bag_input)
            input_shape = bag_input.shape
            # In unbatched bag, expected two dims (n_instance, d_instance)
            if len(input_shape) == 2:
                # Just a single bag on its own, not in a batch, therefore place in a list
                reshaped_input = [bag_input]
                is_unbatched = True
            elif len(input_shape) == 3:
                # Already batched with three dims (n_batch, n_instance, d_instance)
                reshaped_input = bag_input
                is_unbatched = False
            else:
                raise NotImplementedError("Cannot process MIL model input with shape {:}".format(input_shape))
        # Model input is list
        elif isinstance(bag_input, list):
            # Assume input is a list of 2d tensors
            reshaped_input = bag_input
            is_unbatched = False
        # Invalid input type
        else:
            raise ValueError("Invalid model input type {:}".format(type(bag_input)))
        return reshaped_input, is_unbatched

    def _internal_forward(
        self, bags: Union[torch.Tensor, List[torch.Tensor]], bag_instance_positions: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Actually call the network.

        :param bags: Batch of bags.
        :param bag_instance_positions: Batch of instance positions for each bag.
        :return: Dictionary of the network outputs.
        """
        # Stack list of bags to a single tensor
        #  Assumes all bags are the same size
        # Otherwise the input is assumed to be a 3D tensor (already stacked).
        if isinstance(bags, list):
            bags = torch.stack(bags)
        bags = bags.to(self.device)
        # Reshape to match (n_batch, d_instance, n_instance)
        #  d_instance is number of channels
        # bags = bags.transpose(1, 2)
        # Pass through network
        return self.net(bags, bag_instance_positions)

    def get_predictions(
            self,
            args,
            dataset: MILTSCDataset,
            criterion: Callable = cross_entropy_criterion,
    ) -> Dict:
        # Iterate through data loader and gather preds and targets
        all_bag_logits_list = []
        all_targets_list = []
        self.net.eval()
        # Don't need to worry about batch size being too big during evaluation (only training)
        dataloader = dataset.create_dataloader(batch_size=64, shuffle=True)
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                self.net.eval()

                bags = batch["bags"]
                targets = batch["targets"].to(self.device)
                model_out = self(bags)
                bag_logits = model_out["bag_logits"]

                loss = criterion(bag_logits, targets)
                bag_loss = loss
                total_loss += bag_loss

                all_bag_logits_list.append(torch.sigmoid(bag_logits).cpu().numpy())
                # all_bag_logits_list.append(bag_logits.cpu().numpy())
                all_targets_list.append(targets.cpu().numpy())

        # Gather bag logits and targets into tensors
        all_targets = np.vstack(all_targets_list)
        all_bag_logits = np.vstack(all_bag_logits_list)

        # Get probas from logits
        all_pred_probas = np.exp(all_bag_logits) / np.sum(np.exp(all_bag_logits), axis=1, keepdims=True)

        # If in binary case, reduce probas to single prediction (not doing so breaks some of the evaluation metrics)
        if all_pred_probas.shape[1] == 2:
            all_pred_probas = all_pred_probas[:, 1]

        # Get the actual predicted classes
        # _, all_pred_clzs = torch.max(all_bag_logits, dim=1)
        all_pred_clzs = np.argmax(all_bag_logits, axis=1)
        all_targets = np.argmax(all_targets, axis=1)

        return {
            "all_pred_probas": all_pred_probas,
            "all_targets": all_targets
        }

    def __call__(
        self, bag_input: Union[torch.Tensor, List[torch.Tensor]], bag_instance_positions: Optional[torch.Tensor] = None
    ) -> Dict:
        return self.forward(bag_input, bag_instance_positions=bag_instance_positions)


