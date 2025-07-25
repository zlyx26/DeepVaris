import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from DeepVaris.function import *
import time
from DeepVaris.load_data import *
from DeepVaris.model import *
import os
import pandas as pd
import re


def main():
    start_time = time.time()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    k = 10
    batch_size = 32
    samples_num = 10000
    set_feature_num = 784
    important_feature_num = 64
    num_classes = 2
    num_block = 1
    Lambda_list = [0.3,0.05,0.01,0.005,0.001]
    simulation_iterations = 2
    num_epochs_cnn = 5
    num_epochs_mask_list = [20,20,20,35,45]
    num_epochs_dnn = 10
    learning_rate_cnn = 0.0001
    learning_rate_mask = 0.00002
    learning_rate_dnn = 0.01
    cnn_l1_penalty_rate = 0
    cnn_l2_penalty_rate = 0
    dnn_l1_penalty_rate = 0
    dnn_l2_penalty_rate = 0
    Epochs = 50
    dataset_file_name=f'dataset1_{samples_num}_{set_feature_num}'
    for Epoch in range(Epochs):
        set_seed(Epoch)
        if not os.path.exists(
                f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_results/{dataset_file_name}_temp_{Epoch}/'):
            os.makedirs(
                f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_results/{dataset_file_name}_temp_{Epoch}/')
        train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader, val_dataloader, test_dataloader, train_samples_num, val_samples_num, test_samples_num, art, Mask_art, p0, feature_num, non_constant_features = generate_dataset_1(
            samples_num, set_feature_num, batch_size, k, important_feature_num, True)
        random_index = torch.tensor(Mask_art)
        with open(
                f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_results/important_feature_index_{Epoch}.txt',
                'w') as f:
            f.write(f'significant feature index:{random_index.flatten().tolist()}')

        cleanup()
        train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn = get_mask_data(train_X, train_Y, val_X, val_Y,
                                                                                      test_X, test_Y)
        CD_net = Classification_dnn(p0, num_classes)

        _, dnn_initial_accuracy, dnn_initial_loss_p, dnn_initial_auc, dnn_initial_pr, dnn_initial_f1 = CCP(
            train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn,
            CD_net, device, num_epochs_dnn, learning_rate_dnn, dnn_l1_penalty_rate, dnn_l2_penalty_rate, num_classes)
        for i in range(simulation_iterations):
            dataset_name = f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_results/{dataset_file_name}_temp_{Epoch}/{dataset_file_name}_{i}_'
            initial_txt(dataset_name)

            cleanup()
            cc_net = Classification_cnn1(feature_num, k, num_classes)

            cnn_net, cnn_initial_accuracy, cnn_initial_loss_p, auc, pr, f1 = CCP(
                train_dataloader, val_dataloader, test_dataloader,
                cc_net, device, num_epochs_cnn, learning_rate_cnn, cnn_l1_penalty_rate, cnn_l2_penalty_rate, num_classes)

            for j in range(len(Lambda_list)):
                Lambda = Lambda_list[j]
                num_epochs_mask = num_epochs_mask_list[j]
                cleanup()
                Mask_net = Mask_Network1(num_block, feature_num, k)
                select_feature_index, select_feature_num, select_important_num, fdr, cnn_final_loss_p, cnn_final_accuracy,_ = CFS(
                    train_dataloader,test_dataloader,train_samples_num,Mask_net, cnn_net, device, num_epochs_mask, batch_size, Lambda,
                    k, learning_rate_mask, feature_num,random_index)

                cnn_non_if_loss_p, cnn_non_if_accuracy = get_non_if_resuts_classify(test_dataloader, cnn_net, device,
                                                              select_feature_index.flatten().tolist())

                if select_feature_num != 0:
                    train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn = get_mask_data(train_X, train_Y,val_X,val_Y,test_X, test_Y,
                                                                                                  select_feature_index.flatten().tolist(),non_constant_features)
                    cd_net = Classification_dnn(int(select_feature_index.size(3)), num_classes)
                    _, dnn_final_accuracy, dnn_final_loss_p, dnn_final_auc, dnn_final_pr, dnn_final_f1 = CCP(
                        train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn,
                        cd_net, device, num_epochs_dnn, learning_rate_dnn, dnn_l1_penalty_rate, dnn_l2_penalty_rate, num_classes)
                    cleanup()

                    sc_rs(Epoch, dataset_name, cnn_initial_accuracy, dnn_initial_accuracy, cnn_final_accuracy,
                          dnn_final_accuracy, cnn_non_if_accuracy, cnn_initial_loss_p, dnn_initial_loss_p,
                          cnn_final_loss_p, dnn_final_loss_p, cnn_non_if_loss_p,
                          select_feature_num, select_important_num, fdr, select_feature_index.flatten().tolist(),
                          dnn_initial_auc, dnn_initial_pr, dnn_initial_f1, dnn_final_auc, dnn_final_pr, dnn_final_f1)
                else:
                    sc_rs(Epoch, dataset_name, cnn_initial_accuracy, dnn_initial_accuracy, cnn_final_accuracy,
                          0, cnn_non_if_accuracy, cnn_initial_loss_p, dnn_initial_loss_p, cnn_final_loss_p, 0,
                          cnn_non_if_loss_p, select_feature_num, select_important_num, 1, [0], 0, 0, 0, 0, 0, 0)

        dataset_name = f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_'
        initial_txt(dataset_name)
        cleanup()
        Feature_index_num_list = np.zeros((len(Lambda_list), simulation_iterations))
        Imp_feature_index_num_list = np.zeros((len(Lambda_list), simulation_iterations))
        for idx in range(simulation_iterations):
            Final_index_list = []
            for i in range(idx + 1):
                with open(
                        f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_results/{dataset_file_name}_temp_{Epoch}/{dataset_file_name}_{i}_results.txt',
                        'r') as file:
                    data = file.read()
                sfi = re.findall(r'select_feature_index:\[([^\]]+)\]', data)
                feature_index_list = []
                for match in sfi:
                    feature_indices = [int(x) for x in match.split(', ')]
                    feature_index_list.append(feature_indices)
                final_index_list = feature_index_list
                Final_index_list.append(final_index_list)

            for num in range(len(Lambda_list)):
                feature_index_num, feature_index = get_index(Final_index_list, num)
                Feature_index_num_list[num, idx] = feature_index_num
                imp_ft_num = len(set(random_index.flatten().tolist()) & set(feature_index[num]))
                Imp_feature_index_num_list[num, idx] = imp_ft_num
        Feature_index_num_list = Feature_index_num_list[:, 1:]
        Imp_feature_index_num_list = Imp_feature_index_num_list[:, 1:]

        select_feature_num_pd = pd.DataFrame(Feature_index_num_list)
        select_feature_num_pd.to_csv(
            f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_results/{dataset_file_name}_feature_num_{Epoch}.csv',
            index=False, header=False)
        select_important_num_pd = pd.DataFrame(Imp_feature_index_num_list)
        select_important_num_pd.to_csv(
            f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_results/{dataset_file_name}_important_num_{Epoch}.csv',
            index=False, header=False)
        for i in range(len(feature_index)):
            select_feature_num = Feature_index_num_list[i, int(len(Feature_index_num_list[0]) - 1)]
            select_important_num = Imp_feature_index_num_list[i, int(len(Imp_feature_index_num_list[0]) - 1)]
            if select_feature_num == 0:
                sc_rs(Epoch, dataset_name, cnn_initial_accuracy, dnn_initial_accuracy, 0,
                      0, cnn_initial_accuracy,cnn_initial_accuracy, cnn_initial_loss_p, dnn_initial_loss_p, 0, 0, 0, 0, 0, 1, [0],
                      0, 0, 0, 0, 0, 0)
            else:

                fdr = (select_feature_num - select_important_num) / select_feature_num

                cnn_non_if_loss_p, cnn_non_if_accuracy = get_non_if_resuts_classify(test_dataloader, cnn_net, device, feature_index[i])

                train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn = get_mask_data(train_X, train_Y, val_X,val_Y,
                                                                                              test_X, test_Y,feature_index[i],
                                                                                              non_constant_features)
                cd_net = Classification_dnn(int(select_feature_num), num_classes)
                _, dnn_final_accuracy, dnn_final_loss_p, dnn_final_auc, dnn_final_pr, dnn_final_f1 = CCP(
                    train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn,
                    cd_net, device, num_epochs_dnn, learning_rate_dnn, dnn_l1_penalty_rate, dnn_l2_penalty_rate, num_classes)
                cleanup()
                sc_rs(Epoch, dataset_name, cnn_initial_accuracy, dnn_initial_accuracy, cnn_final_accuracy,
                      dnn_final_accuracy, cnn_non_if_accuracy, cnn_initial_loss_p, dnn_initial_loss_p, cnn_final_loss_p,dnn_final_loss_p, cnn_non_if_loss_p,
                      int(select_feature_num), int(select_important_num), fdr, feature_index[i], dnn_initial_auc,
                      dnn_initial_pr, dnn_initial_f1, dnn_final_auc, dnn_final_pr, dnn_final_f1)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")


if __name__ == "__main__":
    main()

