import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to show Images
def show_images(images, titles, suptitle):
    fig, ax = plt.subplots(1, len(images), figsize=(20, 5))
    fig.suptitle(suptitle, size=18, fontweight='bold', y=1)
    for k in range(len(images)):
        img = images[k] / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        ax[k].imshow(np.transpose(npimg, (1, 2, 0)))
        ax[k].set_title(titles[k])
        ax[k].axis('off')
    plt.show();


# Function to count the number of Images per class
def count_images_per_class(dataset, class_names):
    count_dict = {class_name: 0 for class_name in class_names}
    for _, label in dataset:
        count_dict[class_names[label]] += 1
    return count_dict

# Function to print dataset information
def print_dataset_info(dataset_name, count_dict):
    print(f"{dataset_name} Data:")
    total_images = sum(count_dict.values())
    for class_name, count in count_dict.items():
        print(f"Class '{class_name}': {count} images")
    print(f"\tTotal images in {dataset_name.lower()} data: {total_images}")


# Function to plot bar chart for image counts
def plot_image_counts(dataset_counts, dataset_name, color_palette):
    class_names = list(dataset_counts.keys())
    counts = list(dataset_counts.values())
    
    num_classes = len(class_names)
    colors = color_palette(np.linspace(0, 1, num_classes))  # Generate colors from the colormap
    
    x = np.arange(num_classes)
    width = 0.3

    bars = plt.bar(x - width/2, counts, width, color=colors)

    plt.xlabel('Class ---->', fontsize=16)
    plt.ylabel('Number of Images --->', fontsize=16)
    plt.title(f'Distribution of Labels in {dataset_name} Dataset', fontsize=18)
    plt.xticks(x, class_names, rotation=45, ha='right')
    
    # Add counts on top of each bar
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{count}', ha='center', va='bottom', fontsize=12)
    


# Function to plot Training and Validation accuracy and loss
def plot_training_validation(epochs, train_acc_history, val_acc_history, train_loss_history, val_loss_history):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    fig.text(s='Epochs vs. Training and Validation Accuracy/Loss', size=18, fontweight='bold', y=0.95, x=0.28, alpha=0.8)

    colors_dark = sns.color_palette("dark", 8)
    colors_green = sns.color_palette("Greens", 8)
    colors_red = sns.color_palette("Reds", 8)

    sns.despine()
    ax[0].plot(epochs, train_acc_history, marker='o', markerfacecolor=colors_green[2], color=colors_green[3], label='Training Accuracy')
    ax[0].plot(epochs, val_acc_history, marker='o', markerfacecolor=colors_red[2], color=colors_red[3], label='Validation Accuracy')
    ax[0].legend(frameon=False)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')

    sns.despine()
    ax[1].plot(epochs, train_loss_history, marker='o', markerfacecolor=colors_green[2], color=colors_green[3], label='Training Loss')
    ax[1].plot(epochs, val_loss_history, marker='o', markerfacecolor=colors_red[2], color=colors_red[3], label='Validation Loss')
    ax[1].legend(frameon=False)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Training & Validation Loss')

    plt.show();


# Function to plot Confusion matrix Heatmap
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='PiYG', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels',fontsize=16)
    plt.ylabel('True Labels',fontsize=16)
    plt.title('Confusion Matrix',fontsize=20,color='blue')
    plt.show();
