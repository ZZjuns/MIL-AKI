import pandas as pd
import torch
import xgboost as xgb
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, recall_score, \
    precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from millet.util import *

sequence_length = 48

num_features = 10
num_classes = 2  # 二分类
# dataset = "org_yz_48_48"
# dataset = "eicu_48_48"
dataset = "ydy_48_48"

# 加载训练数据
# path = "E:ZZJ/MILTimeSeriesClassification-master/data/patients/aki_samples_48_zscore_6_demo_train.ts"
# path = f'E:\\ZZJ\\MILTimeSeriesClassification-master\\data\\patients\\{dataset}_train.ts'
# samples , labels = load_ts_file(path)
# x_train = torch.stack(samples)
# y_train = labels
#
# # 加载测试数据
# # path = "E:/ZZJ/MILTimeSeriesClassification-master/data/patients/aki_samples_48_zscore_6_demo_test.ts"
# path = f'E:\\ZZJ\\MILTimeSeriesClassification-master\\data\\patients\\{dataset}_test.ts'
# samples2 , labels2 = load_ts_file(path)
# x_test = torch.stack(samples2)
# y_test = labels2

# path = f'E:\\ZZJ\\MILTimeSeriesClassification-master\\data\\patients\\{dataset}.ts'
# samples2 , labels2 = load_ts_file(path)
# x = torch.stack(samples2)
# # y = labels2
# y = torch.stack(labels2)
#
# # 展平时间序列数据
# # X_train = x_train.reshape(x_train.shape[0], -1)  # 变为 (num_samples, sequence_length * num_features)
# # X_test = x_test.reshape(x_test.shape[0], -1)
#
# X = x.reshape(x.shape[0], -1)  # 变为 (num_samples, sequence_length * num_features)

# 定义模型参数
model_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': 5,
    'max_depth': 3,
    'learning_rate': 0.001,
    'random_state': 42
}

# 交叉验证
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold_results = []

k=0
if k == 1:
    # 交叉验证训练
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        print(f"Fold {fold}:")
        # 使用张量索引
        X_val, X_train = X[train_idx], X[val_idx]
        y_val, y_train = y[train_idx], y[val_idx]

        # 转换为 NumPy 数组以供 XGBoost 使用
        X_train = X_train.numpy()
        X_val = X_val.numpy()
        y_train = y_train.numpy()
        y_val = y_val.numpy()
        eval_set = [(X_train, y_train), (X_val, y_val)]
        # 创建模型
        model = XGBClassifier(**model_params)
        model.fit(X_train, y_train,eval_set=eval_set,verbose=True)
        # 预测
        all_pred_probas = model.predict_proba(X_val)
        all_pred_clzs = np.argmax(all_pred_probas, axis=1)
        all_pred_probas = all_pred_probas[:, 1]
        # all_targets = np.argmax(y_val, axis=1)

        # Compute metrics
        acc = accuracy_score(y_val, all_pred_clzs)
        bal_acc = balanced_accuracy_score(y_val, all_pred_clzs)

        roc_auc_ovo_marco = roc_auc_score(y_val, all_pred_probas, average='macro', multi_class='ovo')
        # roc_auc_ovo_marco = 0
        # roc_auc_ovo_micro = roc_auc_score(all_targets,all_pred_probas,average='micro',multi_class='ovo')
        roc_auc_ovo_micro = 0
        roc_auc_ovr_marco = roc_auc_score(y_val, all_pred_probas, average='macro', multi_class='ovr')
        # roc_auc_ovr_marco = 0
        # roc_auc_ovr_micro = roc_auc_score(all_targets, all_pred_probas, average='micro', multi_class='ovr')
        roc_auc_ovr_micro = 0
        # conf_mat = torch.as_tensor(confusion_matrix(all_targets, all_pred_probas), dtype=torch.float)

        f1_marco = f1_score(y_val, all_pred_clzs, average='macro')
        f1_micro = f1_score(y_val, all_pred_clzs, average='micro')

        r_marco = recall_score(y_val, all_pred_clzs, average='macro')
        r_micro = recall_score(y_val, all_pred_clzs, average='micro')

        # 计算AOPRC
        precision, recall, thresholds = precision_recall_curve(y_val, all_pred_probas)
        AOPRC = auc(recall, precision)

        all_results = {
            "acc": acc,
            "roc_auc_ovo_marco": roc_auc_ovo_marco,
            "roc_auc_ovo_micro": roc_auc_ovo_micro,
            "roc_auc_ovr_marco": roc_auc_ovr_marco,
            "roc_auc_ovr_micro": roc_auc_ovr_micro,
            "f1_marco": f1_marco,
            "f1_micro": f1_micro,
            "r_marco": r_marco,
            "r_micro": r_micro,
            "aoprc": AOPRC,
            "bal_acc": bal_acc,
            # "conf_mat": conf_mat,
        }
        fold_results.append(all_results)

        print(
            '\r fold [%d/%d]  accuracy: %.4f  bal. average score: %.4f  roc_auc ovo marco: %.4f  roc_auc ovr marco: %.4f  f1_marco:%.4f  recall:%.4f  aoprc:%.4f' %
            (fold, 5, all_results['acc'], all_results['bal_acc'], all_results['roc_auc_ovo_marco'],all_results['roc_auc_ovr_marco'],all_results['f1_marco'], all_results['r_marco'], all_results['aoprc']))

        save_dir = "../../../results/Xgboost/"
        maybe_mkdir_p(join(save_dir, f'{dataset}'))
        save_dir = make_dirs(join(save_dir, f'{dataset}'))
        maybe_mkdir_p(save_dir)

        save_path = join(save_dir, 'weights')
        os.makedirs(save_path, exist_ok=True)
        save_name = os.path.join(save_path, f'model{fold}.json')
        model.save_model(save_name)

        # plot log loss
        # retrieve performance metrics
        results = model.evals_result()
        epochs = len(results['validation_0']['logloss'])
        x_axis = range(0, epochs)
        fig1, ax = pyplot.subplots()
        ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
        ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
        ax.legend()
        pyplot.ylabel('Log Loss')
        pyplot.title('XGBoost Log Loss')
        pyplot.show()


    avg_acc = sum([result['acc'] for result in fold_results]) / len(fold_results)
    avg_auc = sum([result['roc_auc_ovo_marco'] for result in fold_results]) / len(fold_results)
    avg_f1 = sum([result['f1_marco'] for result in fold_results]) / len(fold_results)
    avg_recall = sum([result['r_marco'] for result in fold_results]) / len(fold_results)
    avg_aoprc = sum([result['aoprc'] for result in fold_results]) / len(fold_results)

    save_dir = "../../../results/Xgboost/"
    # 设置保存路径
    save_dir = os.path.join(save_dir, dataset)
    os.makedirs(save_dir, exist_ok=True)
    csv_file_path = os.path.join(save_dir, 'fold_results.csv')

    # 保存结果为 CSV 文件
    # 构造结果 DataFrame
    fold_results_df = pd.DataFrame(fold_results)
    avg_results = {
        'acc': avg_acc,
        'roc_auc_ovo_marco': avg_auc,
        'f1_marco': avg_f1,
        'r_marco': avg_recall,
        'auprc': avg_aoprc
    }
    avg_results_df = pd.DataFrame([avg_results])
    # 每一折结果和平均值
    fold_results_df = pd.concat([fold_results_df, avg_results_df], ignore_index=True)
    # 设置保存路径
    os.makedirs(save_dir, exist_ok=True)
    csv_file_path = os.path.join(save_dir, 'fold_results.csv')

    # 保存为 CSV 文件
    fold_results_df.to_csv(csv_file_path, index=False)
    print(f"Results saved to {csv_file_path}")
else:
    path = f'E:\\ZZJ\\MILTimeSeriesClassification-master\\data\\patients\\{dataset}_test.ts'
    samples4, labels4 = load_ts_file(path)
    x_train = torch.stack(samples4)
    y_train = torch.stack(labels4)

    path = f'E:\\ZZJ\\MILTimeSeriesClassification-master\\data\\patients\\{dataset}_train.ts'
    samples3, labels3 = load_ts_file(path)
    x_val = torch.stack(samples3)
    y_val = torch.stack(labels3)

    X_train = x_train.reshape(x_train.shape[0], -1)  # 变为 (num_samples, sequence_length * num_features)
    X_val = x_val.reshape(x_val.shape[0], -1)  # 变为 (num_samples, sequence_length * num_features)

    # 转换为 NumPy 数组以供 XGBoost 使用
    X_train = X_train.numpy()
    X_val = X_val.numpy()
    y_train = y_train.numpy()
    y_val = y_val.numpy()
    eval_set = [(X_train, y_train), (X_val, y_val)]
    # 创建模型
    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

    # 预测
    all_pred_probas = model.predict_proba(X_val)
    all_pred_clzs = np.argmax(all_pred_probas, axis=1)
    all_pred_probas = all_pred_probas[:, 1]
    # all_targets = np.argmax(y_val, axis=1)

    # Compute metrics
    acc = accuracy_score(y_val, all_pred_clzs)
    bal_acc = balanced_accuracy_score(y_val, all_pred_clzs)

    roc_auc_ovo_marco = roc_auc_score(y_val, all_pred_probas, average='macro', multi_class='ovo')
    # roc_auc_ovo_marco = 0
    # roc_auc_ovo_micro = roc_auc_score(all_targets,all_pred_probas,average='micro',multi_class='ovo')
    roc_auc_ovo_micro = 0
    roc_auc_ovr_marco = roc_auc_score(y_val, all_pred_probas, average='macro', multi_class='ovr')
    # roc_auc_ovr_marco = 0
    # roc_auc_ovr_micro = roc_auc_score(all_targets, all_pred_probas, average='micro', multi_class='ovr')
    roc_auc_ovr_micro = 0
    # conf_mat = torch.as_tensor(confusion_matrix(all_targets, all_pred_probas), dtype=torch.float)

    f1_marco = f1_score(y_val, all_pred_clzs, average='macro')
    f1_micro = f1_score(y_val, all_pred_clzs, average='micro')

    r_marco = recall_score(y_val, all_pred_clzs, average='macro')
    r_micro = recall_score(y_val, all_pred_clzs, average='micro')

    # 计算AOPRC
    precision, recall, thresholds = precision_recall_curve(y_val, all_pred_probas)
    AOPRC = auc(recall, precision)

    all_results = {
        "acc": acc,
        "roc_auc_ovo_marco": roc_auc_ovo_marco,
        "roc_auc_ovo_micro": roc_auc_ovo_micro,
        "roc_auc_ovr_marco": roc_auc_ovr_marco,
        "roc_auc_ovr_micro": roc_auc_ovr_micro,
        "f1_marco": f1_marco,
        "f1_micro": f1_micro,
        "r_marco": r_marco,
        "r_micro": r_micro,
        "aoprc": AOPRC,
        "bal_acc": bal_acc,
        # "conf_mat": conf_mat,
    }
    fold_results.append(all_results)

    print(
        '\r accuracy: %.4f  bal. average score: %.4f  roc_auc ovo marco: %.4f  roc_auc ovr marco: %.4f  f1_marco:%.4f  recall:%.4f  aoprc:%.4f' %
        (all_results['acc'], all_results['bal_acc'], all_results['roc_auc_ovo_marco'],
         all_results['roc_auc_ovr_marco'], all_results['f1_marco'], all_results['r_marco'], all_results['aoprc']))

    save_dir = "../../../results/Xgboost/"
    maybe_mkdir_p(join(save_dir, f'{dataset}'))
    save_dir = make_dirs(join(save_dir, f'{dataset}'))
    maybe_mkdir_p(save_dir)

    save_path = join(save_dir, 'weights')
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, f'model.json')
    model.save_model(save_name)

    # 保存结果为 CSV 文件
    best_dict = pd.DataFrame([all_results])
    csv_file_path = os.path.join(save_dir, 'best_dict.csv')
    # 检查文件是否存在
    if not os.path.isfile(csv_file_path):
        # 如果文件不存在，则保存为新的 CSV 文件（包含 header）
        best_dict.to_csv(csv_file_path, index=False)
    else:
        # 如果文件存在，则追加数据（不写入 header）
        best_dict.to_csv(csv_file_path, mode='a', index=False, header=False)
    print("save best dict to" + csv_file_path)

# print("---------------------最优模型----------------------------")
# model_best_params = grid_search.best_params_
# model = grid_search.best_estimator_
# 训练
# model.fit(X, y)
# # 预测
# y_pred = model.predict(X_test)
# # 获取预测概率（取第二列为阳性类概率）
# y_pred_proba = model.predict_proba(X_test)[:, 1]
#
# acc = accuracy_score(y_test, y_pred)
#
# # 计算准确率
# acc = accuracy_score(y_test, y_pred)
# bal_acc = balanced_accuracy_score(y_test, y_pred)
# roc_auc_ovo_marco = roc_auc_score(y_test, y_pred_proba, average='macro', multi_class='ovo')
# # roc_auc_ovo_micro = roc_auc_score(all_targets,all_pred_probas,average='micro',multi_class='ovo')
# roc_auc_ovo_micro = 0
# roc_auc_ovr_marco = roc_auc_score(y_test, y_pred_proba, average='macro', multi_class='ovr')
# # roc_auc_ovr_micro = roc_auc_score(all_targets, all_pred_probas, average='micro', multi_class='ovr')
# roc_auc_ovr_micro = 0
# # conf_mat = torch.as_tensor(confusion_matrix(all_targets, all_pred_probas), dtype=torch.float)
#
# # Return results in dict
# all_results = {
#     "loss": 0,
#     "acc": acc,
#     "roc_auc_ovo_marco": roc_auc_ovo_marco,
#     "roc_auc_ovo_micro": roc_auc_ovo_micro,
#     "roc_auc_ovr_marco": roc_auc_ovr_marco,
#     "roc_auc_ovr_micro": roc_auc_ovr_micro,
#     "bal_acc": bal_acc,
#     # "conf_mat": conf_mat,
# }
# print("最优结果:{}".format(all_results['roc_auc_ovo_marco']))
#
#
# save_dir = "../../../results/Xgboost/"
# maybe_mkdir_p(join(save_dir, f'{dataset}'))
# save_dir = make_dirs(join(save_dir, f'{dataset}'))
# maybe_mkdir_p(save_dir)
#
# save_path = join(save_dir, 'weights')
# os.makedirs(save_path, exist_ok=True)
# save_name = os.path.join(save_path, 'model.json')
# model.save_model(save_name)
#
# # 保存结果为 CSV 文件
# best_dict = pd.DataFrame([all_results])
# csv_file_path = os.path.join(save_dir, 'best_dict.csv')
# # 检查文件是否存在
# if not os.path.isfile(csv_file_path):
#     # 如果文件不存在，则保存为新的 CSV 文件（包含 header）
#     best_dict.to_csv(csv_file_path, index=False)
# else:
#     # 如果文件存在，则追加数据（不写入 header）
#     best_dict.to_csv(csv_file_path, mode='a', index=False, header=False)
# print("save best dict to" + csv_file_path)

