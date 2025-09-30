import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

#from Model_3_old import EPOCHS

# --- Common Configuration ---
SEED = 1
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)
device = torch.device("cuda" if cuda else "cpu")

torch.manual_seed(SEED)
if cuda:
    torch.cuda.manual_seed(SEED)

dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# --- Data Transformations (can be customized per model) ---
def get_transforms(model_name="default"):
    if model_name == "Model_3":
        train_transforms = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else: # Default transforms for Model_1 and Model_2
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return train_transforms, test_transforms

# --- Common Data Loading ---
def get_dataloaders(model_name):
    train_transforms, test_transforms = get_transforms(model_name)
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)
    return train_loader, test_loader

# --- Common Training Function ---
def train(model, device, train_loader, optimizer, epoch, train_losses, train_acc):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    y_pred = model(data)
    loss = F.nll_loss(y_pred, target)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    pred = y_pred.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)
    pbar.set_description(desc= f'Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

# --- Common Test Function ---
def test(model, device, test_loader, test_losses, test_acc):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc.append(100. * correct / len(test_loader.dataset))

# --- Plotting Function ---
def plot_results(model_name, train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    fig.suptitle(f'{model_name} Training and Test Results', fontsize=16)
    
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    
    # Adjusting for potential difference in list length if only some batches are stored
    # Using a simpler plot for illustration, can be enhanced for more granular control
    axs[1, 0].plot([acc for i, acc in enumerate(train_acc) if i % 100 == 0]) # Plot every 100th accuracy for better visualization
    axs[1, 0].set_title("Training Accuracy (sampled)")
    
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{model_name}_results.png')
    plt.show()

# --- Main Training Orchestration ---
def run_model_training(model_class, model_name, epochs=15, lr=0.01, momentum=0.9, weight_decay=0, scheduler_type='None'):
    print(f"\n--- Starting training for {model_name} ---")

    model = model_class().to(device)
    summary(model, input_size=(1, 28, 28))

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    scheduler = None
    if scheduler_type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
    
    train_loader, test_loader = get_dataloaders(model_name)

    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(epochs):
        print(f"EPOCH: {epoch}")
        train(model, device, train_loader, optimizer, epoch, train_losses, train_acc)        
        test(model, device, test_loader, test_losses, test_acc)
        if scheduler:
            scheduler.step()
    
    print(f"--- Finished training for {model_name} ---")
    plot_results(model_name, train_losses, train_acc, test_losses, test_acc)

if __name__ == "__main__":
    from Model_1 import Net as Net1
    from Model_2 import Net as Net2
    from Model_3 import Net as Net3 # Assuming Model_3 has been fixed and Net3 is importable

    print("Starting master training script...")

    # Run Model 1
    #run_model_training(Net1, "Model_1", epochs=15, lr=0.01, momentum=0.9)

    # Run Model 2
    #run_model_training(Net2, "Model_2", epochs=15, lr=0.01, momentum=0.9)

    # Run Model 3
    # Model_3 used dropout and CosineAnnealingLR previously, so configure accordingly
    run_model_training(Net3, "Model_3", epochs=15, lr=0.09, momentum=0.9, weight_decay=1e-5, scheduler_type='CosineAnnealingLR')

    print("\nMaster training script finished.")
