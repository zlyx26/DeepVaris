import gc
import tensorflow as tf
from DeepVaris.load_data import *
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np
from scipy.stats import pearsonr

# Function to clean up memory by releasing GPU memory and performing garbage collection
def cleanup():
    torch.cuda.empty_cache()
    gc.collect()
# Function to set the random seed for reproducibility in training and data generation
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.set_random_seed(seed)
# Function to apply mask for feature selection during training
def apply_mask_t(Mask_net, inputs, k):

    mask_model_output = Mask_net(inputs)
    important_scores_vectors = torch.sigmoid(mask_model_output)
    mean_important_scores_vectors = important_scores_vectors.mean(dim=0, keepdim=True)
    binary_vector = torch.where(mean_important_scores_vectors > 0.5, 1.0, 0.0)
    binary_vector = mean_important_scores_vectors + (binary_vector - mean_important_scores_vectors).detach()
    return binary_vector, important_scores_vectors, mean_important_scores_vectors
# Function to apply mask for validation/testing data
def apply_mask_vt(Mask_net, inputs, k):
    mask_model_output = Mask_net(inputs)
    important_scores_vectors = torch.sigmoid(mask_model_output)
    binary_vector = torch.where(important_scores_vectors > 0.5, 1.0, 0.0)
    binary_vector = binary_vector.repeat(1, 1, int(k), 1)
    return binary_vector,important_scores_vectors

# Function to prepare datasets with optional feature selection based on non-constant features or specified indices
def get_mask_data(train_X,train_Y, val_X,val_Y,test_X,test_Y,select_feature_index=None,non_constant_features=None):

    if non_constant_features is not None:
        train_X = train_X[:, non_constant_features]
        val_X = val_X[:, non_constant_features]
        test_X = test_X[:, non_constant_features]
    # Apply feature selection by specified indices if provided
    if select_feature_index is not None:
        indices = np.array(select_feature_index)

        train_X = train_X[:, indices]
        val_X = val_X[:, indices]
        test_X = test_X[:, indices]

    train_dataset = TensorDataset(torch.tensor(train_X), torch.tensor(train_Y).squeeze(-1))
    val_dataset = TensorDataset(torch.tensor(val_X), torch.tensor(val_Y).squeeze(-1))
    test_dataset = TensorDataset(torch.tensor(test_X), torch.tensor(test_Y).squeeze(-1))
    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=50, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader


#Function to initialize a results text file
def initial_txt(dataset_name):
    with open(f'{dataset_name}results.txt', 'w') as f:
        f.write(f'{dataset_name} Simulation Results\n')

def sc_rs(seed,dataset_name,cnn_initial_accuracy,dnn_initial_accuracy,cnn_final_accuracy,dnn_final_accuracy,cnn_non_if_accuracy,
                      cnn_initial_loss_p,dnn_initial_loss_p,cnn_final_loss_p,dnn_final_loss_p,cnn_non_if_loss_p,
                      select_feature_num, select_important_num, fdr,select_feature_index,initial_auc,initial_pr,initial_f1,final_auc,final_pr,final_f1,all_feature_num_list=None,important_feature_num_list=None):
    with open(f'{dataset_name}results.txt', 'a') as f:
        f.write(f'\n Data Seed: {seed} \n')
        f.write(f'Iteration {seed + 1} correlation:\n')
        f.write(f'CNN Initial Accuracy:{cnn_initial_accuracy} \n')
        f.write(f'DNN Initial Accuracy:{dnn_initial_accuracy}\n')
        f.write(f"CNN Final Accuracy:{cnn_final_accuracy}\n")
        f.write(f"DNN Final Accuracy:{dnn_final_accuracy}\n")
        f.write(f'CNN Initial Loss p:{cnn_initial_loss_p}\n')
        f.write(f'DNN Initial Loss p:{dnn_initial_loss_p}\n')
        f.write(f'CNN Final Loss p:{cnn_final_loss_p}\n')
        f.write(f'DNN Final Loss p:{dnn_final_loss_p}\n')
        f.write(f'CNN non if Loss p:{cnn_non_if_loss_p}\n')
        f.write(f"CNN non if accuracy:{cnn_non_if_accuracy}\n")
        f.write(f'# of feature:{select_feature_num}\n')
        f.write(f'# of significant feature:{select_important_num}\n')
        f.write(f'FDR:{fdr}\n')
        f.write(f'select_feature_index:{select_feature_index}\n')
        f.write(f'DNN Initial AUC:{initial_auc}\n')
        f.write(f'DNN Initial PR:{initial_pr}\n')
        f.write(f'DNN Initial F1:{initial_f1}\n')
        f.write(f'DNN Final AUC:{final_auc}\n')
        f.write(f'DNN Final PR:{final_pr}\n')
        f.write(f'DNN Final F1:{final_f1}\n')
        if all_feature_num_list is not None:
            f.write(f'all_feature_num_list:{all_feature_num_list}\n')
        if important_feature_num_list is not None:
            f.write(f'important_feature_num_list:{important_feature_num_list}\n')

def sr_rs(seed, dataset_name, cnn_initial_loss_p, dnn_initial_loss_p, cnn_initial_correlation,dnn_initial_correlation,cnn_final_loss_p, dnn_final_loss_p,cnn_final_correlation,dnn_final_correlation, cnn_non_if_loss_p,cnn_non_if_correlation,
                      select_feature_num, select_important_num, fdr, select_feature_index):
    with open(f'{dataset_name}results.txt', 'a') as f:
        f.write(f'\n Data Seed: {seed} \n')
        f.write(f'CNN Initial Loss p:{cnn_initial_loss_p}\n')
        f.write(f'DNN Initial Loss p:{dnn_initial_loss_p}\n')
        f.write(f'CNN Initial Correlation:{cnn_initial_correlation}\n')
        f.write(f'DNN Initial Correlation:{dnn_initial_correlation}\n')
        f.write(f'CNN Final Loss p:{cnn_final_loss_p}\n')
        f.write(f'DNN Final Loss p:{dnn_final_loss_p}\n')
        f.write(f'CNN Final Correlation:{cnn_final_correlation}\n')
        f.write(f'DNN Final Correlation:{dnn_final_correlation}\n')
        f.write(f'CNN non if Loss p:{cnn_non_if_loss_p}\n')
        f.write(f'CNN non if Correlation:{cnn_non_if_correlation}\n')
        f.write(f'# of feature:{select_feature_num}\n')
        f.write(f'# of significant feature:{select_important_num}\n')
        f.write(f'FDR:{fdr}\n')
        f.write(f'select_feature_index:{select_feature_index}\n')

def rc_rs(seed,dataset_name,cnn_initial_accuracy,dnn_initial_accuracy,cnn_final_accuracy,dnn_final_accuracy,cnn_non_if_accuracy,
                      cnn_initial_loss_p,dnn_initial_loss_p,cnn_final_loss_p,dnn_final_loss_p,cnn_non_if_loss_p,
                      select_feature_num,select_feature_index):
    with open(f'{dataset_name}results.txt', 'a') as f:
        f.write(f'\n Data Seed: {seed} \n')
        f.write(f'CNN Initial Accuracy:{cnn_initial_accuracy} \n')
        f.write(f'DNN Initial Accuracy:{dnn_initial_accuracy}\n')
        f.write(f"CNN Final Accuracy:{cnn_final_accuracy}\n")
        f.write(f"DNN Final Accuracy:{dnn_final_accuracy}\n")
        f.write(f'CNN Initial Loss p:{cnn_initial_loss_p}\n')
        f.write(f'DNN Initial Loss p:{dnn_initial_loss_p}\n')
        f.write(f'CNN Final Loss p:{cnn_final_loss_p}\n')
        f.write(f'DNN Final Loss p:{dnn_final_loss_p}\n')
        f.write(f'CNN non if Loss p:{cnn_non_if_loss_p}\n')
        f.write(f"CNN non if accuracy:{cnn_non_if_accuracy}\n")
        f.write(f'# of feature:{select_feature_num}\n')
        f.write(f'select_feature_index:{select_feature_index}\n')

def rr_rs(seed, dataset_name, cnn_initial_loss_p, dnn_initial_loss_p,cnn_initial_correlation,dnn_initial_correlation, cnn_final_loss_p, dnn_final_loss_p,
                      cnn_final_correlation,dnn_final_correlation,cnn_non_if_loss_p,cnn_non_if_correlation,
                      select_feature_num, select_feature_index):
    with open(f'{dataset_name}results.txt', 'a') as f:
        f.write(f'\n Data Seed: {seed} \n')
        f.write(f'CNN Initial Loss p:{cnn_initial_loss_p}\n')
        f.write(f'DNN Initial Loss p:{dnn_initial_loss_p}\n')
        f.write(f'CNN Initial Correlation:{cnn_initial_correlation}\n')
        f.write(f'DNN Initial Correlation:{dnn_initial_correlation}\n')
        f.write(f'CNN Final Loss p:{cnn_final_loss_p}\n')
        f.write(f'DNN Final Loss p:{dnn_final_loss_p}\n')
        f.write(f'CNN Final Correlation:{cnn_final_correlation}\n')
        f.write(f'DNN Final Correlation:{dnn_final_correlation}\n')
        f.write(f'CNN non if Loss p:{cnn_non_if_loss_p}\n')
        f.write(f'CNN non if Correlation:{cnn_non_if_correlation}\n')
        f.write(f'# of feature:{select_feature_num}\n')
        f.write(f'select_feature_index:{select_feature_index}\n')

# Function for training and evaluating the model (CNN_Classification_Prediction)
def CCP(
        train_dataloader, val_dataloader, test_dataloader,
        model, device, num_epochs, learning_rate, l1_penalty_rate, l2_penalty_rate, num_classes):
    model = model.to(device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_penalty_rate)
    train_loss_list = []

    for epoch in range(num_epochs):
        model.train()
        for train_inputs, train_labels in train_dataloader:
            train_inputs, train_labels = train_inputs.to(device).float(), train_labels.to(device)
            train_labels = torch.argmax(train_labels, dim=1)
            optimizer.zero_grad()
            train_outputs = model(train_inputs)
            train_loss = loss_fun(train_outputs, train_labels)

            if l1_penalty_rate > 0:
                l1_penalty = sum(torch.sum(abs(param)) for param in model.parameters() if param.requires_grad)
                train_loss += l1_penalty_rate * l1_penalty
            train_loss.backward()
            optimizer.step()
            train_loss_list.append(train_loss.item())

        if (epoch + 1) % 50 == 0:
            model.eval()
            val_loss_sum = 0.0
            val_predictions_list, val_labels_list = [], []
            with torch.no_grad():
                for val_inputs, val_labels in val_dataloader:
                    val_inputs, val_labels = val_inputs.to(device).float(), val_labels.to(device)
                    val_outputs = model(val_inputs)
                    val_predictions_list.append(val_outputs)
                    val_labels_list.append(val_labels)
                    val_loss = loss_fun(val_outputs, torch.argmax(val_labels, dim=1))
                    val_loss_sum += val_loss.item()

            val_loss = val_loss_sum / len(val_dataloader)
            val_predictions = torch.cat(val_predictions_list).cpu().numpy()
            val_labels = torch.cat(val_labels_list).cpu().numpy()
            val_predictions = np.argmax(val_predictions, axis=1)
            val_labels = np.argmax(val_labels, axis=1)
            accuracy = np.mean(val_predictions == val_labels)
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss_list[-1]}, Val Loss: {val_loss}, Val Acc: {accuracy}")

    model.eval()
    test_predictions_list, test_labels_list = [], []
    loss_p = 0
    with torch.no_grad():
        for test_inputs, test_labels in test_dataloader:
            test_inputs, test_labels = test_inputs.to(device).float(), test_labels.to(device)
            test_outputs = model(test_inputs)
            test_predictions_list.append(test_outputs)
            test_labels_list.append(test_labels)
            loss_p += loss_fun(test_outputs, torch.argmax(test_labels, dim=1))

    loss_p = loss_p / len(test_dataloader)
    test_predictions = torch.cat(test_predictions_list).cpu().numpy()
    test_labels = torch.cat(test_labels_list).cpu().numpy()
    test_probs = torch.softmax(torch.tensor(test_predictions), dim=1).numpy()
    test_labels = np.argmax(test_labels, axis=1)


    accuracy = np.mean(np.argmax(test_probs, axis=1) == test_labels)
    print("Test accuracy:", accuracy)
    if num_classes == 2:
        auc = roc_auc_score(test_labels, test_probs[:, 1])
        pr = average_precision_score(test_labels, test_probs[:, 1])
        f1 = f1_score(test_labels, np.argmax(test_probs, axis=1))
    else:
        auc = roc_auc_score(test_labels, test_probs, average='macro', multi_class='ovr')  
    

        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(test_labels, classes=np.arange(num_classes))
        pr = average_precision_score(y_true_bin, test_probs, average='macro')
        f1 = f1_score(test_labels, np.argmax(test_probs, axis=1), average='macro')
    return model, accuracy, loss_p,auc,pr,f1

# Function for training and evaluating the model (CNN_Regression_Prediction)
def CRP(train_dataloader, val_dataloader, test_dataloader,
        model, device, num_epochs, learning_rate, l1_penalty_rate, l2_penalty_rate):

    model = model.to(device)
    loss_fun = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_penalty_rate)
    train_loss_list = []

    for epoch in range(num_epochs):
        model.train()
        for train_inputs, train_labels in train_dataloader:
            train_inputs, train_labels = train_inputs.to(device).float(), train_labels.to(device).float()
            optimizer.zero_grad()
            train_outputs = model(train_inputs)
            train_loss = loss_fun(train_outputs, train_labels)
            if l1_penalty_rate > 0:
                l1_penalty = sum(torch.sum(abs(param)) for param in model.parameters() if param.requires_grad)
                train_loss += l1_penalty_rate * l1_penalty
            train_loss.backward()
            optimizer.step()
            train_loss_list.append(train_loss.item())


        if (epoch + 1) % 50 == 0:
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for val_inputs, val_labels in val_dataloader:
                    val_inputs, val_labels = val_inputs.to(device).float(), val_labels.to(device).float()
                    val_outputs = model(val_inputs)
                    val_loss = loss_fun(val_outputs, val_labels)
                    val_loss_sum += val_loss.item()

            avg_val_loss = val_loss_sum / len(val_dataloader)
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss_list[-1]}, Val Loss: {avg_val_loss}")

    model.eval()
    cleanup()
    model = model.to(device)
    mse = 0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for test_inputs, test_labels in test_dataloader:
            test_inputs, test_labels = test_inputs.to(device).float(), test_labels.to(device).float()
            test_outputs = model(test_inputs)
            mse += loss_fun(test_outputs, test_labels)
            all_outputs.append(test_outputs.cpu().numpy())
            all_labels.append(test_labels.cpu().numpy())


    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    correlation, _ = pearsonr(all_outputs.flatten(), all_labels.flatten())
    mse = mse / len(test_dataloader)
    print("Test MSE:", mse)
    return model, mse,correlation

# Function for feature selection during training using MaskNet(Classification_Feature_Selection)
def CFS(
        train_dataloader,  test_dataloader,train_samples_num,
        Mask_net, cnn, device, num_epochs, batch_size, Lambda, k,
        learning_rate, feature_num, random_index=None,feature_num_list=None):

    Mask_net  = Mask_net.to(device)
    cnn = cnn.to(device)
    print_counter = math.ceil(train_samples_num // batch_size)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(Mask_net.parameters(), lr=learning_rate)
    all_feature_num_list=[]
    important_feature_num_list=[]
    for epoch in range(num_epochs):
        print(f"-------Epoch {epoch + 1} -------")
        Mask_net.train()
        iteration = 0
        for train_inputs, train_labels in train_dataloader:
            iteration += 1
            train_inputs, train_labels = train_inputs.to(device).float(), train_labels.to(device)
            optimizer.zero_grad()
            size = train_inputs.size(0)

            non_mask_predicted_result = cnn(train_inputs)
            non_mask_predicted_result = torch.argmax(non_mask_predicted_result, dim=1)
            binary_vector, important_scores_vectors, mean_important_scores_vectors = apply_mask_t(Mask_net,
                                                                                                     train_inputs, k)
            mask_predicted_result = cnn(train_inputs*(binary_vector.repeat(1, 1, int(k), 1)).float())
            # Calculate loss for different components
            loss_c1, n = 0, important_scores_vectors.size(0)
            for i in range(n):
                for j in range(i + 1, n):
                    difference = torch.abs(important_scores_vectors[i] - important_scores_vectors[j])
                    loss_c1 += difference.sum()
            loss_c1 = loss_c1 / ((n * (n - 1)) / 2)
            loss_c2 = ((important_scores_vectors - mean_important_scores_vectors) ** 2).mean(dim=0, keepdim=True).sum()
            loss_p = loss_fun(mask_predicted_result, non_mask_predicted_result)
            loss_s = Lambda * torch.sum(binary_vector) / (size)
            train_loss = loss_p + loss_s + loss_c1 + loss_c2
            train_loss.backward()
            optimizer.step()

            if iteration == print_counter:
                print(f"Train Cross Entropy Loss: {loss_p}")
                print(f"Train Ls Loss: {loss_s}")
                print(f"Train Total Loss: {train_loss}")
                print(f"Train loss_c1: {loss_c1}")
                print(f"Train loss_c2: {loss_c2}")
                print(f"Train # of original variables: {torch.sum(binary_vector)}")
                if random_index is not None:
                    print(
                        f"Train, # of significant variables: {torch.sum(binary_vector[:, :, :, random_index])}")
                    if feature_num_list is not None:
                        a=torch.sum(binary_vector) / size
                        b=torch.sum(binary_vector[:, :, :, random_index]) / size
                        all_feature_num_list.append([epoch,a.item()])
                        important_feature_num_list.append([epoch,b.item()])

    Mask_net.eval()
    with torch.no_grad():
        loss_p= 0
        test_predictions_list, test_labels_list = [], []
        for test_inputs, test_labels in test_dataloader:
            test_inputs,test_labels = test_inputs.to(device).float(),test_labels.to(device)
            binary_vector,important_scores_vectors = apply_mask_vt(Mask_net, test_inputs, k)
            mask_predicted_result = cnn(binary_vector*test_inputs.float())
            loss_p += loss_fun(mask_predicted_result, torch.argmax(test_labels, dim=1))
            test_predictions_list.append(mask_predicted_result)
            test_labels_list.append(test_labels)
        loss_p = loss_p / len(test_dataloader)
        test_predictions = torch.cat(test_predictions_list)
        test_labels = torch.cat(test_labels_list)
        test_predictions = test_predictions.cpu().numpy()
        test_labels = test_labels.cpu().numpy()
        test_labels = np.argmax(test_labels, axis=1)
        accuracy = np.mean(np.argmax(test_predictions, axis=1) == test_labels)



    Mask_net.eval()
    cnn.eval()
    zero_tensor = torch.zeros(1, 1, 1, feature_num).to(device)
    Count=0
    zero_misv = torch.zeros(1, 1, 1, feature_num).to(device)
    with torch.no_grad():
        for train_inputs, train_labels in train_dataloader:
            train_inputs, train_labels = train_inputs.to(device).float(), train_labels.to(device)
            binary_vector,important_scores_vectors = apply_mask_vt(Mask_net, train_inputs, k)
            zero_tensor += binary_vector.sum(dim=2, keepdim=True).mean(dim=0, keepdim=True)/k
            misv = important_scores_vectors.mean(dim=0, keepdim=True)
            zero_misv+=misv
            Count+=1
    zero_misv=zero_misv/Count
    prob_tensor = zero_tensor / len(train_dataloader)
    feature_index = torch.zeros((1, 1, 1, prob_tensor.size(3)), dtype=torch.long)
    for i in range(prob_tensor.size(3)):
        if prob_tensor[0, 0, 0, i] == 1:
            feature_index[0, 0, 0, i] = i
    select_feature_num = torch.count_nonzero(feature_index)
    if random_index is not None:
        select_important_num = torch.count_nonzero(feature_index[:, :, :, random_index])
    feature_index = torch.nonzero(feature_index)
    select_feature_index = torch.zeros((1, 1, 1, feature_index.size(0)), dtype=torch.long)

    for i in range(feature_index.size(0)):
        select_feature_index[0, 0, 0, i] = feature_index[i, 3]


    if random_index is not None:
        if select_feature_num > 0:
            fdr = (select_feature_num - select_important_num) / select_feature_num
            if feature_num_list is not None:
                return select_feature_index, select_feature_num, select_important_num, fdr, loss_p,accuracy,all_feature_num_list,important_feature_num_list,zero_misv
            else:
                return select_feature_index, select_feature_num, select_important_num, fdr, loss_p, accuracy,zero_misv
        else:
            if feature_num_list is not None:
                return select_feature_index, select_feature_num,0,0 ,loss_p,accuracy,all_feature_num_list,important_feature_num_list,zero_misv
            else:
                print('select feature num = 0')
                return select_feature_index, select_feature_num,0,0 ,loss_p,accuracy,zero_misv
    else:
        if feature_num_list is not None:
            return select_feature_index, select_feature_num, loss_p, accuracy,all_feature_num_list,important_feature_num_list,zero_misv
        else:
            return select_feature_index, select_feature_num, loss_p,accuracy,zero_misv

# Function for feature selection during training using MaskNet(Regression_Feature_Selection)
def RFS(
        train_dataloader,  test_dataloader,
        train_samples_num, Mask_net, cnn, device, num_epochs, batch_size, Lambda, k,
        learning_rate, feature_num, random_index=None):

    Mask_net  = Mask_net.to(device)
    cnn = cnn.to(device)
    print_counter = math.ceil(train_samples_num // batch_size)
    loss_fun = nn.MSELoss()
    optimizer = optim.Adam(Mask_net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"-------Epoch {epoch + 1} -------")
        Mask_net.train()
        iteration = 0
        for train_inputs, train_labels in train_dataloader:
            iteration += 1
            train_inputs, train_labels = train_inputs.to(device).float(), train_labels.to(device).float()
            optimizer.zero_grad()
            size = train_inputs.size(0)
            non_mask_predicted_result = cnn(train_inputs)
            binary_vector, important_scores_vectors, mean_important_scores_vectors = apply_mask_t(Mask_net,
                                                                                                     train_inputs, k)

            mask_predicted_result = cnn(train_inputs*(binary_vector.repeat(1, 1, int(k), 1)).float())
            # Calculate loss for different components
            loss_c1, n = 0, important_scores_vectors.size(0)
            for i in range(n):
                for j in range(i + 1, n):
                    difference = torch.abs(important_scores_vectors[i] - important_scores_vectors[j])
                    loss_c1 += difference.sum()
            loss_c1 = loss_c1 / ((n * (n - 1)) / 2)
            loss_c2 = ((important_scores_vectors - mean_important_scores_vectors) ** 2).mean(dim=0, keepdim=True).sum()
            loss_p = loss_fun(mask_predicted_result, non_mask_predicted_result)
            loss_s = Lambda * torch.sum(binary_vector) / (size)
            train_loss = loss_p + loss_s + loss_c1 + loss_c2
            train_loss.backward()
            optimizer.step()

            if iteration == print_counter:
                print(f"Train MSE: {loss_p}")
                print(f"Train Ls Loss: {loss_s}")
                print(f"Train Total Loss: {train_loss}")
                print(f"Train loss_c1: {loss_c1}")
                print(f"Train loss_c2: {loss_c2}")
                print(f"Train # of original variables: {torch.sum(binary_vector)}")
                if random_index is not None:
                    print(
                        f"Train, # of significant variables: {torch.sum(binary_vector[:, :, :, random_index])}")

    Mask_net.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        loss_p= 0
        for test_inputs, test_labels in test_dataloader:
            test_inputs = test_inputs.to(device).float()
            binary_vector,_ = apply_mask_vt(Mask_net, test_inputs, k)
            mask_predicted_result = cnn(test_inputs*binary_vector.float())
            loss_p += loss_fun(mask_predicted_result, test_labels.to(device))
            all_outputs.append(mask_predicted_result.cpu().numpy())
            all_labels.append(test_labels.cpu().numpy())
        loss_p = loss_p / len(test_dataloader)
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    correlation, _ = pearsonr(all_outputs.flatten(), all_labels.flatten())

    Mask_net.eval()
    cnn.eval()
    zero_tensor = torch.zeros(1, 1, 1, feature_num).to(device)
    with torch.no_grad():
        for train_inputs, train_labels in train_dataloader:
            train_inputs, train_labels = train_inputs.to(device).float(), train_labels.to(device).float()
            binary_vector,_ = apply_mask_vt(Mask_net, train_inputs, k)
            zero_tensor += binary_vector.sum(dim=2, keepdim=True).mean(dim=0, keepdim=True)/k


    prob_tensor = zero_tensor / len(train_dataloader)
    feature_index = torch.zeros((1, 1, 1, prob_tensor.size(3)), dtype=torch.long)
    for i in range(prob_tensor.size(3)):
        if prob_tensor[0, 0, 0, i] == 1:
            feature_index[0, 0, 0, i] = i
    select_feature_num = torch.count_nonzero(feature_index)
    if random_index is not None:
        select_important_num = torch.count_nonzero(feature_index[:, :, :, random_index])
    feature_index = torch.nonzero(feature_index)
    select_feature_index = torch.zeros((1, 1, 1, feature_index.size(0)), dtype=torch.long)
    for i in range(feature_index.size(0)):
        select_feature_index[0, 0, 0, i] = feature_index[i, 3]
    if random_index is not None:
        if select_feature_num > 0:
            fdr = (select_feature_num - select_important_num) / select_feature_num
            return select_feature_index, select_feature_num, select_important_num, fdr, loss_p,correlation
        else:
            return select_feature_index, select_feature_num, select_important_num, 0, loss_p,correlation
    else:
        return select_feature_index, select_feature_num,loss_p,correlation


# Function to evaluate the model's performance by masking selected features during testing
def get_non_if_resuts_classify(test_dataloader, model, device, index):
    model = model.to(device)
    loss_fun = nn.CrossEntropyLoss()

    test_predictions_list, test_labels_list = [], []


    loss_p = 0

    # Disable gradient calculation to save memory and computation during testing
    with torch.no_grad():
        for test_inputs, test_labels in test_dataloader:
            test_inputs = test_inputs.to(device).float()
            mask = torch.ones_like(test_inputs).to(device)

            mask[..., index] = 0

            mask_input = test_inputs * mask

            test_outputs = model(mask_input)

            test_predictions_list.append(test_outputs)
            test_labels_list.append(test_labels)

            loss_p += loss_fun(test_outputs, torch.argmax(test_labels.to(device), dim=1))

    # Compute the average loss across all batches
    loss_p = loss_p / len(test_dataloader)

    # Concatenate all the predictions and labels from the entire test set
    test_predictions = torch.cat(test_predictions_list)
    test_labels = torch.cat(test_labels_list)

    # Apply sigmoid to predictions and convert them to NumPy arrays
    test_predictions = torch.sigmoid(test_predictions).cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    # Get the predicted class labels (index of the max probability)
    test_predictions = np.argmax(test_predictions, axis=1)
    test_labels = np.argmax(test_labels, axis=1)

    accuracy = np.mean(test_predictions == test_labels)
    return loss_p, accuracy



def get_non_if_resuts_regression(test_dataloader, model, device, index):

    model = model.to(device)
    loss_fun = nn.MSELoss()
    loss_p = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for test_inputs, test_labels in test_dataloader:
            test_inputs = test_inputs.to(device).float()
            mask = torch.ones_like(test_inputs).to(device)
            mask[..., index] = 0
            mask_input = test_inputs * mask
            test_outputs = model(mask_input)
            loss_p += loss_fun(test_outputs, test_labels.to(device))
            all_outputs.append(test_outputs.cpu().numpy())
            all_labels.append(test_labels.cpu().numpy())


    loss_p = loss_p / len(test_dataloader)
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    correlation, _ = pearsonr(all_outputs.flatten(), all_labels.flatten())

    return loss_p, correlation

def get_index(Final_index_list,num):
    feature_index =[]
    feature_index_num=0
    for i in range(len(Final_index_list[0])):
        index_sub=[]
        for j in range(len(Final_index_list)):
            if j==0:
                index_sub=Final_index_list[j][i]
            else:
                index_sub=list(set(index_sub)&set(Final_index_list[j][i]))
        feature_index.append(index_sub)
        if i==num:
            feature_index_num=len(index_sub)
    return feature_index_num,feature_index