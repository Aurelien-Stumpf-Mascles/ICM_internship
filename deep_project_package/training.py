import torch
from tqdm import tqdm
import os
from deep_project_package.compute_procrustes import compute_procrustes, compute_orthonormal_projection

#Training the model
def train_model(model,trainloader,testloader,device,criterion,epochs=1,optimizer=None,scheduler=None,type='regression',print_epoch=10,save=False,folder_name='model'):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    
    folder_path = os.path.join("./model_weights",folder_name)

    model.train()

    for epoch in range(epochs):

        total_loss = 0
        if type == 'classification':
            correct = 0

        with tqdm(trainloader, total=len(trainloader), unit="batch", desc=f'Epoch {epoch}') as tepoch:
            for X,y in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                X = X.to(device)
                y = y.to(device)
                outputs = model(X)
                loss = criterion(outputs,y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()/len(trainloader.dataset)
                if type == 'regression':  
                    tepoch.set_postfix(loss=total_loss)
                if type == 'classification':
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == y.flatten()).sum().item()/len(trainloader.dataset)
                    tepoch.set_postfix(loss=total_loss, accuracy=correct)
            
            if scheduler is not None:
                scheduler.step(total_loss)

            if epoch % print_epoch == 0:
                print(f'Epoch {epoch}')
                print("lr: ", optimizer.param_groups[0]['lr'])
                if type == 'classification':
                    print("-------------------------")
                    evaluate_classification_model(model,testloader,device,criterion)
                    print("-------------------------")
                else:
                    print("-------------------------")
                    evaluate_regression_model(model,testloader,device,criterion)
                    print("-------------------------")

                if save:
                    torch.save(model.state_dict(), folder_path + '/model_{}.pt'.format(epoch))

#train model with orthogonalization
def train_model_orthogonal(model,trainloader,testloader,device,criterion,epochs=1,optimizer=None,scheduler=None,type='regression'):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for X,y in trainloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = criterion(outputs,y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for layer in model.named_parameters():
                num_layer = int(layer[0][1])
                if 'weight' in layer[0] :
                    W = layer[1]
                    A = W.data.cpu().numpy().T
                    W.data = torch.from_numpy(compute_procrustes(A)).float().t().to(device)
    
        if scheduler is not None:
            scheduler.step(total_loss)   

        if epoch % 10 == 0:
            print(f'Epoch {epoch}')
            print("lr: ", optimizer.param_groups[0]['lr'])
            if type == 'classification':
                print(total_loss)
                print("Training Loss")
                evaluate_classification_model(model,trainloader,device,criterion)
                print("Test Loss")
                evaluate_classification_model(model,testloader,device,criterion)
                print("-------------------------")
            else:
                print("Training Loss")
                print(total_loss / len(trainloader.dataset))
                print("Test Loss")
                evaluate_regression_model(model,testloader,device,criterion)
                print("-------------------------")

#train model with
def train_model_orthonormal(model,trainloader,testloader,device,criterion,epochs=1,optimizer=None,scheduler=None,type='regression'):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for X,y in trainloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = criterion(outputs,y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for layer in model.named_parameters():
            #check if last layer is reached
            layer_num = int(layer[0][1])
            if layer_num == 1 and 'weight' in layer[0]:
                W = layer[1]
                A = W.data.cpu().numpy()
                W.data = torch.from_numpy(compute_orthonormal_projection(A)).float().to(device)
        
        if scheduler is not None:
            scheduler.step(total_loss)   

        if epoch % 10 == 0:
            print(f'Epoch {epoch}')
            print("lr: ", optimizer.param_groups[0]['lr'])
            if type == 'classification':
                print(total_loss)
                print("Training Loss")
                evaluate_classification_model(model,trainloader,device,criterion)
                print("Test Loss")
                evaluate_classification_model(model,testloader,device,criterion)
                print("-------------------------")
            else:
                print("Training Loss")
                print(total_loss / len(trainloader.dataset))
                evaluate_regression_model(model,testloader,device,criterion)
                print("-------------------------")

#Evaluating the model
def evaluate_classification_model(model,dataloader,device,criterion):
    model.eval()
    total_loss = 0
    total = 0
    correct = 0

    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = criterion(outputs,y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y.flatten()).sum().item()

    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    total_loss, correct, len(dataloader.dataset),
    100. * correct / len(dataloader.dataset)))

def evaluate_regression_model(model,dataloader,device,criterion):
    model.eval()
    total_loss = 0
    total = 0

    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = criterion(outputs,y)
            total_loss += loss.item()
            total += y.size(0)

    print('Test set: Avg. loss: {:.4f}'.format(total_loss))