import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
from model import Discriminator, Generator


def display_sample_images(folder_path, epoch_number):
    file_name = f"sample_images_epoch_{epoch_number}.png"
    file_path = os.path.join(folder_path, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Sample image for epoch {epoch_number} not found in {folder_path}.")

    image = Image.open(file_path)

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Sample Images - Epoch {epoch_number}")
    plt.show()


def display_loss_curves(folder_path, epoch_number):
    file_name = f"loss_curves_epoch_{epoch_number}.png"
    file_path = os.path.join(folder_path, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Loss curve image for epoch {epoch_number} not found in {folder_path}.")

    image = Image.open(file_path)

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Loss Curves - Epoch {epoch_number}")
    plt.show()


def load_checkpoint(folder_path, epoch_number):
    file_name = f"gan_checkpoint_epoch_{epoch_number}.pth"
    file_path = os.path.join(folder_path, file_name)
    checkpoint = torch.load(file_path)

    return checkpoint


def load_checkpoint_and_initialize_objects(folder_path, epoch_number, discriminator_config, generator_config, d_optimizer_config, g_optimizer_config, device):
    checkpoint = load_checkpoint(folder_path, epoch_number)

    generator = Generator(**generator_config).to(device)
    discriminator = Discriminator(**discriminator_config).to(device)

    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])

    d_optimizer = optim.Adam(
        discriminator.parameters(),
        **d_optimizer_config
    )
    g_optimizer = optim.Adam(
        generator.parameters(),
        **g_optimizer_config
    )

    d_optimizer.load_state_dict(checkpoint["d_optimizer"])
    g_optimizer.load_state_dict(checkpoint["g_optimizer"])

    checkpoint_objects = {
        "epoch": checkpoint["epoch"],
        "generator": generator,
        "discriminator": discriminator,
        "d_optimizer": d_optimizer,
        "g_optimizer": g_optimizer,
        "fixed_noise": checkpoint["fixed_noise"],
        "d_losses": checkpoint["d_losses"],
        "g_losses": checkpoint["g_losses"],
        "mean_d_losses": checkpoint["mean_d_losses"],
        "mean_g_losses": checkpoint["mean_g_losses"]
    }

    return checkpoint_objects
