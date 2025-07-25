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

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    k=5
    batch_size = 32
    dataset_file_name='her2'
    num_classes = 2
    num_block = 1

    Lambda_list = [0.03,0.02,0.01,0.007,0.005,0.003,0.001]

    num_epochs_cnn = 5

    num_epochs_mask_list = [100,200,200,200,250,300,350]


    num_epochs_dnn = 10
    learning_rate_cnn = 0.0001
    learning_rate_mask = 0.00002
    learning_rate_dnn=0.01
    simulation_iterations = 10
    cnn_l1_penalty_rate = 0
    cnn_l2_penalty_rate = 0
    dnn_l1_penalty_rate = 0
    dnn_l2_penalty_rate = 0
    Epochs = 1
    for Epoch in range(Epochs):
        set_seed(Epoch)
        if not os.path.exists(
                f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_results/{dataset_file_name}_temp_{Epoch}/'):
            os.makedirs(
                f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_results/{dataset_file_name}_temp_{Epoch}/')
        train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader, val_dataloader, test_dataloader, train_samples_num, val_samples_num, test_samples_num, p0, feature_num, non_constant_features = load_her2(
            batch_size, k, True)
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
            cc_net = Classification_cnn2(feature_num, k, num_classes)

            cnn_net, cnn_initial_accuracy, cnn_initial_loss_p, auc, pr, f1 = CCP(
                train_dataloader, val_dataloader, test_dataloader,
                cc_net, device, num_epochs_cnn, learning_rate_cnn, cnn_l1_penalty_rate, cnn_l2_penalty_rate,
                num_classes)

            for j in range(len(Lambda_list)):
                Lambda = Lambda_list[j]
                num_epochs_mask = num_epochs_mask_list[j]
                cleanup()
                Mask_net = Mask_Network3(num_block, feature_num, k)
                select_feature_index, select_feature_num, cnn_final_loss_p, cnn_final_accuracy, misv = CFS(
                    train_dataloader, test_dataloader, train_samples_num, Mask_net, cnn_net, device, num_epochs_mask,
                    batch_size, Lambda, k, learning_rate_mask, feature_num)
                cnn_non_if_loss_p, cnn_non_if_accuracy = get_non_if_resuts_classify(test_dataloader, cnn_net, device,
                                                                                    select_feature_index.flatten().tolist())
                arr = misv.squeeze().cpu().numpy()

                df = pd.DataFrame({
                    'feature_index': np.arange(arr.shape[0]),
                    'mask_value': arr
                })
                df.to_csv(
                    f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_results/{dataset_file_name}_temp_{Epoch}/mask{i}{j}.csv',
                    index=False)
                if select_feature_num != 0:
                    train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn = get_mask_data(train_X, train_Y,
                                                                                                  val_X, val_Y,
                                                                                                  test_X, test_Y,
                                                                                                  select_feature_index.flatten().tolist())
                    cleanup()
                    cd_net = Classification_dnn(int(select_feature_index.size(3)), num_classes)
                    _, dnn_final_accuracy, dnn_final_loss_p, dnn_final_auc, dnn_final_pr, dnn_final_f1 = CCP(
                        train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn,
                        cd_net, device, num_epochs_dnn, learning_rate_dnn, dnn_l1_penalty_rate, dnn_l2_penalty_rate,
                        num_classes)
                    cleanup()

                    rc_rs(Epoch, dataset_name, cnn_initial_accuracy, dnn_initial_accuracy, cnn_final_accuracy,
                          dnn_final_accuracy, cnn_non_if_accuracy, cnn_initial_loss_p, dnn_initial_loss_p,
                          cnn_final_loss_p, dnn_final_loss_p, cnn_non_if_loss_p,
                          select_feature_num, select_feature_index.flatten().tolist()
                          )
                else:
                    rc_rs(Epoch, dataset_name, cnn_initial_accuracy, dnn_initial_accuracy, 0,
                          0, cnn_initial_accuracy, cnn_initial_loss_p, dnn_initial_loss_p,
                          0, 0, cnn_initial_loss_p,
                          0, 0)

        cleanup()
        dataset_name = f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_'
        initial_txt(dataset_name)
        Feature_index_num_list = np.zeros((len(Lambda_list), simulation_iterations))

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
        Feature_index_num_list = Feature_index_num_list[:, 1:]

        select_feature_num_pd = pd.DataFrame(Feature_index_num_list)
        select_feature_num_pd.to_csv(
            f'results/{dataset_file_name}/{dataset_file_name}_{Epoch}_results/{dataset_file_name}_feature_num_{Epoch}.csv',
            index=False, header=False)
        for i in range(len(feature_index)):
            select_feature_num = Feature_index_num_list[i, int(len(Feature_index_num_list[0]) - 1)]
            if select_feature_num == 0:
                rc_rs(Epoch, dataset_name, cnn_initial_accuracy, dnn_initial_accuracy, 0,
                      0, cnn_initial_accuracy, cnn_initial_accuracy, cnn_initial_loss_p, dnn_initial_loss_p, 0, 0, 0,
                      [0])
            else:

                cnn_non_if_loss_p, cnn_non_if_accuracy = get_non_if_resuts_classify(test_dataloader, cnn_net, device,
                                                                                    feature_index[i])
                train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn = get_mask_data(train_X, train_Y, val_X,
                                                                                              val_Y,
                                                                                              test_X, test_Y,
                                                                                              feature_index[i])
                cd_net = Classification_dnn(int(select_feature_num), num_classes)
                _, dnn_final_accuracy, dnn_final_loss_p, dnn_final_auc, dnn_final_pr, dnn_final_f1 = CCP(
                    train_dataloader_dnn, val_dataloader_dnn, test_dataloader_dnn,
                    cd_net, device, num_epochs_dnn, learning_rate_dnn, dnn_l1_penalty_rate, dnn_l2_penalty_rate,
                    num_classes)
                cleanup()

                rc_rs(Epoch, dataset_name, cnn_initial_accuracy, 0, 0,
                      dnn_final_accuracy, cnn_non_if_accuracy,
                      cnn_initial_loss_p, 0, 0, dnn_final_loss_p, cnn_non_if_loss_p,
                      select_feature_num, feature_index[i])

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")


if __name__ == "__main__":
    main()

