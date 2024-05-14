import torch
from package.compute_procrustes import compute_procrustes, compute_orthonormal_projection

#Training the model
def train_model(model,trainloader,testloader,device,criterion,epochs=1,optimizer=None,scheduler=None,type='regression'):
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

#Training the model
def train_model_zero(model,trainloader,testloader,device,criterion,epochs=1,optimizer=None,scheduler=None,type='regression'):
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
        
        if scheduler is not None:
            scheduler.step(total_loss)   

        #put all low weights to zero
        if epoch % 10 == 0:
            for layer in model.named_parameters():
                if 'weight' in layer[0]:
                    param = layer[1] 
                    param.data[param.abs() < 0.3] = 0       

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

#Training the model
def train_model_degree(model,trainloader,testloader,device,criterion,epochs=1,optimizer=None,scheduler=None,type='regression',degree_threshold=0.3):
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
        
        if scheduler is not None:
            scheduler.step(total_loss)   

        #put all weights consecutive to low degree to zero
        if epoch % 100 == 0:
            '''
            remove_nodes = None
            num_layer = 0
            for layer in model.named_parameters():
                if num_layer == 0:
                    num_layer += 1
                    continue
                if 'weight' in layer[0]:
                    param = layer[1]
                    if remove_nodes is not None:
                        param.data[:,remove_nodes] = 0
                    degree = torch.sum(param.abs(),dim=1).detach()
                    remove_nodes = degree < degree_threshold
            '''

            remove_nodes = None
            num_layer = 0
            for layer in reversed(list(model.named_parameters())):
                if num_layer == 0:
                    num_layer += 1
                    continue
                if 'weight' in layer[0]:
                    param = layer[1]
                    if remove_nodes is not None:
                        param.data[remove_nodes,:] = 0
                    degree = torch.sum(param.abs(),dim=0).detach()
                    remove_nodes = degree < degree_threshold 
                     

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
                print("Test Loss")
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

    print(f'Accuracy: {100 * correct / total}')
    print(f'Loss: {total_loss / total}')

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

    print(f'Loss: {total_loss / total}')