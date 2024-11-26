import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch


def remove_dataset_on_disk():
    if os.path.exists('./data'):
        shutil.rmtree('./data')
        print("Removing 'data' folder")
    else:
        print("'data' folder does not exists")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_predictions(img, true_labels, predicted_labels, batch_size, classes):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(batch_size * 2, 2))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

    for i in range(len(true_labels)):
        label_text = f"True: {classes[true_labels[i]]}\nPred: {classes[predicted_labels[i]]}"
        plt.text(i * (npimg.shape[2] / len(true_labels)) + (npimg.shape[2] / (2 * len(true_labels))),
                 -2,
                 label_text,
                 ha='center',
                 va='bottom',
                 fontsize=12,
                 color='white',
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    plt.show()


def display_predictions(model, test_loader, classes, device, num_images=8, num_sample_rows=3):
    dataiter = iter(test_loader)
    for _ in range(num_sample_rows):
        images, labels = next(dataiter)
        images, labels = images[:num_images], labels[:num_images]
        images, labels = images.to(device), labels.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

        images = images.cpu()
        labels = labels.cpu()
        predicted = predicted.cpu()

        show_predictions(torchvision.utils.make_grid(
            images), labels, predicted, num_images, classes)
        for i in range(len(labels)):
            print(
                f"True label: {classes[labels[i]]}, Predicted: {classes[predicted[i]]}")
