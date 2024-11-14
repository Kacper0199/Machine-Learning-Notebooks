import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, feature_maps=[64, 128, 128], dropout_rate=0.2, leaky_relu_neg_slope=0.2, output_image_size=8):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=feature_maps[0], kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=feature_maps[0])

        self.conv2 = nn.Conv2d(
            in_channels=feature_maps[0], out_channels=feature_maps[1], kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=feature_maps[1])

        self.conv3 = nn.Conv2d(
            in_channels=feature_maps[1], out_channels=feature_maps[2], kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=feature_maps[2])

        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_neg_slope)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(
            feature_maps[2] * output_image_size * output_image_size, 1)
        self.activation_fc = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)

        x = torch.flatten(x, start_dim=1)

        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation_fc(x)

        return x


class Generator(nn.Module):
    def __init__(self, noise_dim=128, feature_maps=[128, 256, 512], leaky_relu_neg_slope=0.2, output_channels=3, output_image_size=8):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.fc = nn.Linear(noise_dim, output_image_size *
                            output_image_size * feature_maps[0])

        self.transp_conv1 = nn.ConvTranspose2d(
            in_channels=feature_maps[0], out_channels=feature_maps[0], kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=feature_maps[0])

        self.transp_conv2 = nn.ConvTranspose2d(
            in_channels=feature_maps[0], out_channels=feature_maps[1], kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=feature_maps[1])

        self.transp_conv3 = nn.ConvTranspose2d(
            in_channels=feature_maps[1], out_channels=feature_maps[2], kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=feature_maps[2])

        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_neg_slope)
        self.output_image_size = output_image_size

        self.conv_final = nn.Conv2d(
            in_channels=feature_maps[2], out_channels=output_channels, kernel_size=5, stride=1, padding=2)
        self.activation_final = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)

        # batch size, channels, width, height
        x = x.view(-1, self.noise_dim, self.output_image_size,
                   self.output_image_size)

        x = self.transp_conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.transp_conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.transp_conv3(x)
        x = self.bn3(x)
        x = self.activation(x)

        x = self.conv_final(x)
        x = self.activation_final(x)

        return x


def train_discriminator(discriminator, real_images, fake_images, optimizer, device, shuffle_labels=False):
    real_batch_size = real_images.size(0)
    fake_batch_size = fake_images.size(0)

    # Adding noise to labels - label smoothing (prevents from aggresive discriminator's work)
    smoothed_real_labels = (torch.ones(real_batch_size, 1) * 0.95 +
                            0.05 * torch.rand(real_batch_size, 1)).to(device)
    smoothed_fake_labels = (torch.zeros(fake_batch_size, 1) +
                            0.05 * torch.rand(fake_batch_size, 1)).to(device)

    # Label flipping with 10% probability
    # Skip last batch (real_batch_size == fake_batch_size condition)
    # - sometimes the last batch size of real photos differ from the last batch of fake photos
    if shuffle_labels and real_batch_size == fake_batch_size and torch.rand(1).item() < 0.1:
        # Flip the labels
        real_labels, fake_labels = smoothed_fake_labels, smoothed_real_labels
    else:
        # Keep the labels as they are
        real_labels, fake_labels = smoothed_real_labels, smoothed_fake_labels

    optimizer.zero_grad()

    # Training on real images
    real_predictions = discriminator(real_images)
    real_loss = nn.BCELoss()(real_predictions, real_labels)

    # Training on fake images
    fake_predictions = discriminator(fake_images)
    fake_loss = nn.BCELoss()(fake_predictions, fake_labels)

    # Total loss
    total_loss = real_loss + fake_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def train_generator(generator, discriminator, fake_images, optimizer, device):
    batch_size = fake_images.size(0)
    # We want discriminator to classify fake images as real
    real_labels = torch.ones(batch_size, 1).to(device)

    optimizer.zero_grad()
    predictions = discriminator(fake_images)
    loss = nn.BCELoss()(predictions, real_labels)
    loss.backward()
    optimizer.step()

    return loss.item()


class GeneratorUpsampling(nn.Module):
    def __init__(self, noise_dim=128, leaky_relu_neg_slope=0.2, output_channels=3, output_image_size=8):
        super(GeneratorUpsampling, self).__init__()
        self.noise_dim = noise_dim

        self.fc = nn.Linear(noise_dim, output_image_size *
                            output_image_size * 128)

        # Upsampling blocks
        self.upsample_block1 = self.upsample_block(
            128, 64, leaky_relu_neg_slope)
        self.upsample_block2 = self.upsample_block(
            64, 32, leaky_relu_neg_slope)
        self.upsample_block3 = self.upsample_block(
            32, 32, leaky_relu_neg_slope)

        self.conv_final = nn.Conv2d(
            in_channels=32, out_channels=output_channels, kernel_size=5, stride=1, padding=2)
        self.activation_final = nn.Tanh()

    def upsample_block(self, in_channels, out_channels, leaky_relu_neg_slope):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=leaky_relu_neg_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=leaky_relu_neg_slope, inplace=True)
        )

    def forward(self, x):
        # Upscale the noise vector to 8x8x128
        x = self.fc(x)
        x = x.view(-1, 128, 8, 8)

        # Upsampling blocks
        x = self.upsample_block1(x)
        x = self.upsample_block2(x)
        x = self.upsample_block3(x)

        x = self.conv_final(x)
        x = self.activation_final(x)

        return x
