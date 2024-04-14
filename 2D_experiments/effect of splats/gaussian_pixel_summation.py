import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

def generate_gaussian_splat(mean, cov, color, alpha, x, y):

    inv_cov = torch.inverse(cov)
    pos = torch.stack((x.flatten(), y.flatten()), dim=1)
    diff = pos - mean
    exp_term = -0.5 * (diff @ inv_cov * diff).sum(dim=1)
    gaussian= torch.exp(exp_term)
    gaussian=gaussian.view(x.shape)
    # take transpose
    gaussian=torch.transpose(gaussian, 0, 1)
    return alpha * gaussian * color


def render_image(width, height, splats):
    image = torch.zeros((height, width), device=device)
    for splat in splats:
        # mean, cov, color, alpha = splat
        mean=splat[0:2]
        cov=splat[2:6]
        cov=cov.reshape(2,2)
        color=splat[6:7]
        color=color.squeeze(0)
        alpha=splat[7:8]
        alpha=alpha.squeeze(0)
        x = torch.arange(width, device=device)
        y = torch.arange(height, device=device)
        x, y = torch.meshgrid(x, y)
        contribution = generate_gaussian_splat(mean, cov, color, alpha, x, y)
        image += contribution
    return torch.clamp(image, 0, 1)

def initialize_splats(image, num_splats):
    height, width = image.shape
    splats = []
    for _ in range(num_splats):
        x = torch.randint(0, width, (1,)).item()
        y = torch.randint(0, height, (1,)).item()
        mean = torch.tensor([x, y], device=device)
        cov = torch.eye(2, device=device) * 10
        cov = cov.view(-1)
        color = torch.tensor(image[y, x]).to(device)
        color=color.view(-1)
        alpha = torch.tensor(1.0, device=device)
        alpha=alpha.view(-1).to(device)
        combined= torch.cat((mean, cov, color, alpha))
        splats.append(combined)

    splats = torch.stack(splats).requires_grad_(True)
    return splats

def optimize_splats(image, splats, num_iterations):
    directory = f"results"
    os.makedirs(directory, exist_ok=True)
    height, width= image.shape
    image_tensor = torch.from_numpy(image).to(device)
    optimizer = optim.Adam([splats], lr=1e-1)

    for ep in range(num_iterations):
        optimizer.zero_grad()
        rendered_image = render_image(width, height, splats)
        loss=torch.sum((rendered_image - image_tensor)**2)
        loss.backward()
        optimizer.step()

        # Display the target image and the approximated image
        plt.subplot(1, 2, 1)
        plt.imshow(rendered_image.clone().detach().cpu(),cmap='hot', interpolation='nearest')
        plt.title('Target Image')
        plt.subplot(1, 2, 2)
        plt.imshow(image_tensor.clone().detach().cpu(),cmap='hot', interpolation='nearest')
        plt.title('Approximated Image')
        plt.tight_layout()
        # Create filename
        filename = f'image_{ep}.jpg'

        # Construct the full file path
        file_path = os.path.join(directory, filename)

        plt.savefig(file_path,bbox_inches='tight')

        if ep%10==0:
            print(f'Epoch {ep}, Loss: {loss.item()}')

    return splats


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the target image
target_image = Image.open('ref.jpg')

im_size=128
# reshape
target_image=target_image.resize((im_size,im_size))

if target_image.mode != 'L':
    target_image = target_image.convert('L')


transform = transforms.Compose([
    transforms.ToTensor(),
])
target_image_tensor = transform(target_image).to(device).squeeze(0)


num_splats = 100
num_iterations = 2000

# Initialize the Gaussian splats
splats = initialize_splats(target_image_tensor.cpu().numpy(), num_splats)

# Optimize the Gaussian splats
optimized_splats = optimize_splats(target_image_tensor.cpu().numpy(), splats, num_iterations)

# # Render the approximated image
# approximated_image_tensor = render_image(target_image_tensor.shape[1], target_image_tensor.shape[0], optimized_splats)
# approximated_image = transforms.ToPILImage()(approximated_image_tensor.cpu())

