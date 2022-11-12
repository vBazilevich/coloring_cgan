import enum
import numpy as np
import torch
import torchvision
import dataset
import model
import matplotlib.pyplot as plt

plt.ion()

if __name__ == "__main__":
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    label_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = dataset.ImageColorizationDataset('dataset/', transforms, label_transforms)

    BATCH_SIZE = 64
    dataloader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, drop_last=True, num_workers=6, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = model.Generator().to(device)
    discriminator = model.Discriminator().to(device)

    loss = torch.nn.BCELoss()
    struct_loss = torch.nn.L1Loss()
    d_optim = torch.optim.SGD(discriminator.parameters(), lr=0.0004)
    g_optim = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((64, 64)), vmin=0, vmax=1)

    N_EPOCHS = 100
    INSTANCE_NOISE = 0.03
    for epoch_idx in range(N_EPOCHS):
        G_loss = []
        D_loss = []
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)

            # Generate noise and move it the device
            noise = torch.randn(BATCH_SIZE, 100).to(device)

            # Forward pass         
            generated_data = generator(noise, image)
            
            true_data = label.to(device)
            true_labels = torch.FloatTensor(BATCH_SIZE).uniform_(0.7, 1.2).to(device)
            false_labels = torch.zeros(BATCH_SIZE).to(device)

            # Clear optimizer gradients        
            d_optim.zero_grad()

            # Forward pass with true data as input
            discriminator_output_for_true_data = discriminator(label + torch.randn(*label.shape).to(device) * INSTANCE_NOISE, image).view(BATCH_SIZE)
            # Compute Loss
            true_discriminator_loss = loss(discriminator_output_for_true_data, true_labels)
            # Forward pass with generated data as input
            discriminator_output_for_generated_data = discriminator(generated_data.detach() + torch.randn(*generated_data.shape).to(device) * INSTANCE_NOISE, image).view(BATCH_SIZE)
            # Compute Loss 
            generator_discriminator_loss = loss(
                discriminator_output_for_generated_data, false_labels
            )
            # Average the loss
            discriminator_loss = (
                true_discriminator_loss + generator_discriminator_loss
            ) / 2

            # Backpropagate the losses for Discriminator model      
            discriminator_loss.backward()
            d_optim.step()

            D_loss.append(discriminator_loss.data.item())

            # Track discriminator gradients
            d_total_norm = 0
            for p in discriminator.parameters():
                param_norm = p.grad.detach().data.norm(2)
                d_total_norm += param_norm.item() ** 2
            d_total_norm = d_total_norm ** 0.5


            # Clear optimizer gradients

            g_optim.zero_grad()

            # It's a choice to generate the data again
            generated_data = generator(noise, image)
            # Forward pass with the generated data
            discriminator_output_on_generated_data = discriminator(generated_data + torch.randn(*generated_data.shape).to(device) * INSTANCE_NOISE, image).view(BATCH_SIZE)
            # Compute loss
            generator_loss = loss(discriminator_output_on_generated_data, true_labels) + struct_loss(generated_data, label)


            # Backpropagate losses for Generator model.
            generator_loss.backward()
            # torch.nn.utils.clip_grad_norm_(generator.parameters(), 10)
            g_optim.step()

            G_loss.append(generator_loss.data.item())

            # Track discriminator gradients
            g_total_norm = 0
            for p in generator.parameters():
                param_norm = p.grad.detach().data.norm(2)
                g_total_norm += param_norm.item() ** 2
            g_total_norm = g_total_norm ** 0.5
            print(f'[TRAIN] G: {g_total_norm} D: {d_total_norm}')

        # Evaluate the model
        if ((epoch_idx + 1) % 3 == 0):
            total_samples = 0
            correctly_identified = 0
            with torch.no_grad():
                for idx, (image, label) in enumerate(dataloader):
                    image, label = image.to(device), label.to(device)
                    noise = torch.randn(BATCH_SIZE, 100).to(device)
                    generated_data = generator(noise, image)
                    if idx == 0:
                        vis = generated_data.cpu().view(BATCH_SIZE, 3, 64, 64).permute((0, 2, 3, 1))
                        for x in vis:
                            im.set_data(x.detach().numpy())
                            fig.canvas.flush_events()

                    # Evaluate discriminator accuracy
                    total_samples += 2 * BATCH_SIZE
                    correctly_identified += (discriminator(generated_data, image) <= 0.5).sum().item()\
                            + (discriminator(label, image) >= 0.5).sum().item()

            d_acc = correctly_identified / total_samples
            print(f'Discriminator accuracy: {d_acc}')
        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
                (epoch_idx), N_EPOCHS, torch.mean(torch.FloatTensor(D_loss)), torch.mean(torch.FloatTensor(G_loss))))

