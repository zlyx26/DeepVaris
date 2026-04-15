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
    num_block = 1

    Lambda_list=[5,2,1,0.5,0.2,0.1]
    simulation_iterations = 5
    num_epochs_cnn = 50
    num_epochs_mask_list = [10,10,10,20,20,30]
    num_epochs_dnn = 10

    learning_rate_cnn = 0.0001
    learning_rate_mask = 0.00005
    learning_rate_dnn = 0.01
    cnn_l1_penalty_rate = 0.01
    cnn_l2_penalty_rate = 0
    dnn_l1_penalty_rate = 0
    dnn_l2_penalty_rate = 0
    Epochs = 50
    dataset_file_name = f'dataset5_{samples_num}_{set_feature_num}'
    for Epoch in range(Epochs):
        set_seed(Epoch)
        if not os.path.exists(
                f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_results/{dataset_file_name}_temp_{Epoch}/'):
            os.makedirs(
                f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_results/{dataset_file_name}_temp_{Epoch}/')
        train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader, val_dataloader, test_dataloader, train_samples_num, val_samples_num, test_samples_num, art, Mask_art, p0, feature_num, non_constant_features = generate_dataset_5(
            samples_num, set_feature_num, batch_size, k, important_feature_num,True)
        random_index = torch.tensor(Mask_art)
        with open(
                f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_results/important_feature_index_{Epoch}.txt',
                'w') as f:
            f.write(f'significant feature index:{random_index.flatten().tolist()}')
        cleanup()
        train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn = get_mask_data(train_X, train_Y, val_X,val_Y,
                                                                                      test_X, test_Y)
        RD_net = Regression_dnn(p0)
        _, dnn_initial_loss_p, dnn_initial_correlation = CRP(
            train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn,
            RD_net, device, num_epochs_dnn, learning_rate_dnn, dnn_l1_penalty_rate, dnn_l2_penalty_rate)
        for i in range(simulation_iterations):
            dataset_name = f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_results/{dataset_file_name}_temp_{Epoch}/{dataset_file_name}_{i}_'
            initial_txt(dataset_name)

            cleanup()
            rc_net = Regression_cnn1(feature_num, k)

            cnn_net, cnn_initial_loss_p, cnn_initial_correlation = CRP(
                train_dataloader, val_dataloader, test_dataloader,
                rc_net, device, num_epochs_cnn, learning_rate_cnn, cnn_l1_penalty_rate, cnn_l2_penalty_rate)
            for j in range(len(Lambda_list)):
                Lambda = Lambda_list[j]
                num_epochs_mask = num_epochs_mask_list[j]

                cleanup()

                Mask_net = Mask_Network1(num_block, feature_num, k)
                select_feature_index, select_feature_num, select_important_num, fdr, cnn_final_loss_p, cnn_final_correlation = RFS(
                    train_dataloader, test_dataloader,train_samples_num, Mask_net, cnn_net, device, num_epochs_mask, batch_size, Lambda, k, learning_rate_mask, feature_num,
                    random_index)
                cnn_non_if_loss_p, cnn_non_if_correlation = get_non_if_resuts_regression(test_dataloader, cnn_net, device,
                                                                   select_feature_index.flatten().tolist())
                if select_feature_num != 0:
                    train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn = get_mask_data(train_X, train_Y,val_X,val_Y,
                                                                                                  test_X, test_Y,select_feature_index.flatten().tolist(),non_constant_features)
                    cleanup()
                    rd_net = Regression_dnn(int(select_feature_index.size(3)))
                    _, dnn_final_loss_p, dnn_final_correlation = CRP(
                        train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn,
                        rd_net, device, num_epochs_dnn, learning_rate_dnn, dnn_l1_penalty_rate, dnn_l2_penalty_rate)
                    cleanup()

                    sr_rs(Epoch, dataset_name, cnn_initial_loss_p, dnn_initial_loss_p, cnn_initial_correlation,
                          dnn_initial_correlation, cnn_final_loss_p, dnn_final_loss_p,
                          cnn_final_correlation, dnn_final_correlation, cnn_non_if_loss_p, cnn_non_if_correlation,
                          select_feature_num, select_important_num, fdr, select_feature_index.flatten().tolist())
                else:
                    sr_rs(Epoch, dataset_name, cnn_initial_loss_p, dnn_initial_loss_p, cnn_initial_correlation,
                          dnn_initial_correlation, cnn_final_loss_p, 999,
                          cnn_final_correlation, 0, cnn_non_if_loss_p, cnn_non_if_correlation,
                          select_feature_num, select_important_num, fdr, select_feature_index.flatten().tolist())


        dataset_name = f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_'
        initial_txt(dataset_name)
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
                sr_rs(Epoch, dataset_name, cnn_initial_loss_p, dnn_initial_loss_p, cnn_initial_correlation, dnn_initial_correlation,
                      999, 999, cnn_initial_correlation, 0, cnn_initial_loss_p,
                      0.9, 0, 0, 0,[0])
            else:

                fdr = (select_feature_num - select_important_num) / select_feature_num
                cnn_non_if_loss_p, cnn_non_if_correlation = get_non_if_resuts_regression(test_dataloader, cnn_net, device,
                                                                   feature_index[i])

                train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn = get_mask_data(train_X, train_Y, val_X,
                                                                                              val_Y,
                                                                                              test_X, test_Y,
                                                                                              feature_index[i],
                                                                                              non_constant_features)
                Rd_net = Regression_dnn(int(select_feature_num))
                _, dnn_final_loss_p, dnn_final_correlation = CRP(
                    train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn,
                    Rd_net, device, num_epochs_dnn, learning_rate_dnn, dnn_l1_penalty_rate, dnn_l2_penalty_rate)
                cleanup()

                sr_rs(Epoch, dataset_name, cnn_initial_loss_p, dnn_initial_loss_p, cnn_initial_correlation,
                      dnn_initial_correlation, cnn_final_loss_p, dnn_final_loss_p,
                      cnn_final_correlation, dnn_final_correlation, cnn_non_if_loss_p, cnn_non_if_correlation,
                      int(select_feature_num), int(select_important_num), fdr, feature_index[i])


    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")


if __name__ == "__main__":
    main()

