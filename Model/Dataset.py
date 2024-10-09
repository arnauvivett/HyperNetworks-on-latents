dataset_path = '~/datasets'
root = dataset_path  # You can modify this path if needed

dataset = datasets.MovingMNIST(root=root, download=True, transform=transforms.Compose([
        transforms.Lambda(lambda x: x.float()),  # Convert to float
        transforms.Normalize((0.5,), (1.0,))
    ]))

video_data = torch.stack([torch.stack([img/255+0.002 for img in dataset[i]]) for i in range(len(dataset))])

# Split the tensor into training and validation sets
train_size = int(0.8 * video_data.shape[0])
val_size = video_data.shape[0] - train_size
training_tensor = video_data[:train_size]
validation_tensor = video_data[train_size:]

training_tensor[5,0].max() #max val

#the dataset variance: 
data_variance = torch.var((video_data).flatten(start_dim=1))


# Create TensorDataset objects for training and validation sets
train_dataset = TensorDataset(training_tensor)
val_dataset = TensorDataset(validation_tensor)

# Create DataLoader objects for training and validation sets
training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Adjust batch_size as needed
validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)  # No need to shuffle validation data


# Visualize the data
for i,(data,) in enumerate(training_loader):
    print(data.shape)
    if i==0:
        break

def show_video(frames): 
    fig, axs = plt.subplots(2, 10, figsize=(20, 4))
    for i, (image, ax) in enumerate(zip(frames, axs.flat)):
        # Display the image
        ax.imshow(image, cmap='gray')
        ax.axis('off')  # Turn off axis
    fig.tight_layout()
    plt.show()

first_video = data[0, :, 0, :, :]
show_video(first_video)


