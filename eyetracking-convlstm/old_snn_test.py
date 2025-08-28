import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tables
import cv2
import matplotlib.pyplot as plt
import tqdm
from scipy.ndimage import median_filter

import snntorch as snn
from snntorch import surrogate

pretrained = False
test_one = True
height = N = 60
width = M = 80
batch_size = 16
seq = 40
stride = 1
stride_val = 1
chunk_size = 500
num_epochs = 10

# reproducibility
seed = 2
version = 3
torch.manual_seed(seed)
np.random.seed(seed)

log_dir = f'old_snn_log_s{seed}_v{version}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
plot_dir = f'old_snn_plot_s{seed}_v{version}'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


# ---------------- utility functions ----------------

def normalize_data(data):
    img_data = np.array(data)
    mean = np.mean(img_data)
    std = np.std(img_data)
    if std == 0:
        return img_data
    normalized_img_data = (img_data - mean) / (std + 1e-10)
    return normalized_img_data


def create_samples(data, sequence, stride):
    num_samples = data.shape[0]
    chunk_num = num_samples // chunk_size
    chunk_starts = np.arange(chunk_num) * chunk_size
    within_chunk_indices = np.arange(sequence) + np.arange(0, chunk_size - sequence + 1, stride)[:, None]
    indices = chunk_starts[:, None, None] + within_chunk_indices[None, :, :]
    indices = indices.reshape(-1, indices.shape[-1])
    subframes = data[indices]
    return subframes


# ---------------- Dataset ----------------
class EventDataset(Dataset):
    def __init__(self, folder, target_dir, seq, stride):
        self.folder = sorted(folder)
        self.target_dir = target_dir
        self.seq = seq
        self.stride = stride
        self.target = self._concatenate_files()
        self.interval = int((chunk_size - self.seq) / self.stride + 1)

    def __len__(self):
        return len(self.folder) * self.interval

    def __getitem__(self, index):
        file_index = index // self.interval
        sample_index = index % self.interval

        file_path = self.folder[file_index]
        with tables.open_file(file_path, 'r') as file:
            sample = file.root.vector[sample_index]
            sample_resize = []
            for i in range(len(sample)):
                sample_resize.append(normalize_data(cv2.resize(sample[i, 0], (int(width), int(height)))))
            sample_resize = np.expand_dims(np.array(sample_resize), axis=1)  # (seq, 1, H, W)

        label1 = self.target[index][:, 0] / M
        label2 = self.target[index][:, 1] / N
        label = np.concatenate([label1.reshape(-1, 1), label2.reshape(-1, 1)], axis=1)

        return torch.from_numpy(sample_resize).float(), torch.from_numpy(label).float()

    def _concatenate_files(self):
        sorted_target_file_paths = sorted(self.target_dir)
        target = []
        for file_path in sorted_target_file_paths:
            with open(file_path, 'r') as target_file:
                lines = target_file.readlines()
                lines = lines[3::4]
            lines = [list(map(float, line.strip().split())) for line in lines]
            target.extend(lines)
        targets = np.array(torch.tensor(target))
        extended_labels = create_samples(targets, self.seq, self.stride)
        return torch.from_numpy(extended_labels)


def load_filenames(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_paths(base_dir, names, ext):
    out = []
    for name in names:
        if os.path.isabs(name):
            p = name
            if not p.lower().endswith(ext):
                p = p + ext
        else:
            p = os.path.join(base_dir, name + ext)
        out.append(os.path.normpath(p))
    return out


data_dir_train = os.path.join(ROOT_DIR, "DATA", "pupil_st", "data_ts_pro", "train")
data_dir_val = os.path.join(ROOT_DIR, "DATA", "pupil_st", "data_ts_pro", "val")

target_dir = os.path.join(ROOT_DIR, "DATA", "pupil_st", "label")

train_filenames = load_filenames('train_files.txt')
val_filenames = load_filenames('val_files.txt')

data_train = make_paths(data_dir_train, train_filenames, '.h5')
data_val = make_paths(data_dir_val, val_filenames, '.h5')
target_train = make_paths(target_dir, train_filenames, '.txt')
target_val = make_paths(target_dir, val_filenames, '.txt')

train_dataset = EventDataset(data_train, target_train, seq, stride)
val_dataset = EventDataset(data_val, target_val, seq, stride_val)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
valid_dataloader_plt = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class MySNNModel(nn.Module):
    def __init__(self, height, width, in_channels=1, beta=0.9):
        super().__init__()
        self.height = height
        self.width = width

        # conv feature extractor
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.lif1 = snn.Leaky(beta=beta, reset_mechanism='subtract')
        self.lif2 = snn.Leaky(beta=beta, reset_mechanism='subtract')
        self.lif3 = snn.Leaky(beta=beta, reset_mechanism='subtract')

        self.bn_gates = nn.BatchNorm2d(32)

        # compute flattened dimension after two poolings
        fh = height // 4
        fw = width // 4
        flat_dim = 32 * fh * fw

        self.fc1 = nn.Linear(flat_dim, 128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

        self.mem1 = None
        self.mem2 = None
        self.mem3 = None

    def reset_hidden(self):
        self.mem1 = None
        self.mem2 = None
        self.mem3 = None

    def detach_hidden(self):
        if self.mem1 is not None:
            self.mem1 = self.mem1.detach()
        if self.mem2 is not None:
            self.mem2 = self.mem2.detach()
        if self.mem3 is not None:
            self.mem3 = self.mem3.detach()

    def forward(self, x):
        b, c, seq, h, w = x.shape
        device = x.device

        outputs = []

        for t in range(seq):
            xt = x[:, :, t, :, :]

            z1 = self.conv1(xt)
            spk1, self.mem1 = self.lif1(z1, self.mem1)
            p1 = self.pool(spk1)

            z2 = self.conv2(p1)
            spk2, self.mem2 = self.lif2(z2, self.mem2)
            p2 = self.pool(spk2)

            z3 = self.conv3(p2)
            z3 = self.bn_gates(z3)
            spk3, self.mem3 = self.lif3(z3, self.mem3)

            read = self.mem3.view(b, -1) if isinstance(self.mem3, torch.Tensor) else spk3.view(b, -1)

            x_fc = F.relu(self.fc1(read))
            x_fc = self.drop(x_fc)
            out = self.fc2(x_fc)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=0).permute(1, 0, 2)
        return outputs


# instantiate model
input_dim = 1
model = MySNNModel(height, width, in_channels=input_dim, beta=0.9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# params, criterion, optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

start_epoch = 0

if pretrained:
    ckpt_path = (f'checkpoint_snn_v{version}.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)

# ---------------- Training loop ----------------

best_val_loss = float('inf')

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    total_batches = len(train_dataloader)

    for batch_idx, data in tqdm.tqdm(enumerate(train_dataloader, 0), total=total_batches):
        images, targets = data
        images = images.permute(0, 2, 1, 3, 4).to(device).float()
        targets = targets.to(device).float()

        model.reset_hidden()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)

        loss = criterion(outputs, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.detach_hidden()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    epoch_loss = running_loss / max(1, total_batches)

    # ---------------- Validation ----------------
    model.eval()
    val_running_loss = 0.0
    num_values = 0
    num_values_3 = 0
    num_values_5 = 0
    num_values_1 = 0
    tot_values = 0

    with torch.no_grad():
        for images, targets in valid_dataloader:
            images = images.permute(0, 2, 1, 3, 4).to(device).float()
            targets = targets.to(device).float()

            model.reset_hidden()

            outputs = model(images)
            val_loss = criterion(outputs, targets)
            val_running_loss += val_loss.item()

            dis = targets - outputs
            dis[:, :, 0] *= height
            dis[:, :, 1] *= width
            dist = torch.norm(dis, dim=-1)

            num_values = num_values + torch.sum(dist > 10)
            num_values_5 = num_values_5 + torch.sum(dist > 5)
            num_values_3 = num_values_3 + torch.sum(dist > 3)
            num_values_1 = num_values_1 + torch.sum(dist > 1)
            tot_values = tot_values + dist.numel()

        val_epoch_loss = val_running_loss / max(1, len(valid_dataloader))
        err_rate = num_values / tot_values if tot_values > 0 else torch.tensor(0.)
        err_rate_3 = num_values_3 / tot_values if tot_values > 0 else torch.tensor(0.)
        err_rate_5 = num_values_5 / tot_values if tot_values > 0 else torch.tensor(0.)
        err_rate_1 = num_values_1 / tot_values if tot_values > 0 else torch.tensor(0.)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")
        print(
            f"Error rates - >1px: {err_rate_1:.4f}, >3px: {err_rate_3:.4f}, >5px: {err_rate_5:.4f}, >10px: {err_rate:.4f}")

        # Log results
        file_path = os.path.join(log_dir, 'training_log.txt')
        with open(file_path, 'a') as f:
            f.write(
                f"Size {height}, Epoch {epoch}, Loss: {val_epoch_loss}, err_rate_1:{err_rate_1}, err_rate_3:{err_rate_3}, err_rate_5:{err_rate_5}  err: {err_rate} num_values: {num_values} tot_values: {tot_values}\n")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': float(val_epoch_loss),
            }, 'checkpoint_snn.pth')

    valid_dataloader_plt = DataLoader(val_dataset, batch_size=100, shuffle=False)

    for t_plot, data in enumerate(valid_dataloader_plt):
        if t_plot == 1:
            break
        images, targets = data
        frames_plot = images.permute(0, 2, 1, 3, 4).to(device).float()
        target_plot = targets.to(device).float()

        t_l_list = []
        t_r_list = []
        o_l_list = []
        o_r_list = []

        with torch.no_grad():
            for i in range(len(frames_plot)):
                model.reset_hidden()

                img = frames_plot[i].unsqueeze(0)
                outputs_plot = model(img)
                out_np = outputs_plot.detach().cpu().numpy()
                t_l = target_plot[i][:, 0].cpu().numpy().flatten()
                t_r = target_plot[i][:, 1].cpu().numpy().flatten()
                o_l = out_np[:, :, 0].flatten()
                o_r = out_np[:, :, 1].flatten()
                t_l_list.append(t_l)
                t_r_list.append(t_r)
                o_l_list.append(o_l)
                o_r_list.append(o_r)

        t_l_numpy = np.array(t_l_list).flatten()
        t_r_numpy = np.array(t_r_list).flatten()
        o_l_numpy = np.array(o_l_list).flatten()
        o_r_numpy = np.array(o_r_list).flatten()

        # Enhanced plotting from the new implementation
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        # Plot X coordinates (iris position x)
        ax1.plot(t_l_numpy, label='target x', linewidth=2, alpha=0.8)
        ax1.plot(o_l_numpy, label='predicted x', linewidth=2, alpha=0.8, linestyle='--')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Iris X Position (normalized)')
        ax1.set_title('Iris X Position: Target vs Predicted')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        # Dynamic y-axis based on data range
        x_range = np.concatenate([t_l_numpy, o_l_numpy])
        if len(x_range) > 0 and np.ptp(x_range) > 0:
            ax1.set_ylim([max(0, np.min(x_range) - 0.05), min(1, np.max(x_range) + 0.05)])

        # Plot Y coordinates (iris position y)
        ax2.plot(t_r_numpy, label='target y', linewidth=2, alpha=0.8)
        ax2.plot(o_r_numpy, label='predicted y', linewidth=2, alpha=0.8, linestyle='--')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Iris Y Position (normalized)')
        ax2.set_title('Iris Y Position: Target vs Predicted')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        # Dynamic y-axis based on data range
        y_range = np.concatenate([t_r_numpy, o_r_numpy])
        if len(y_range) > 0 and np.ptp(y_range) > 0:
            ax2.set_ylim([max(0, np.min(y_range) - 0.05), min(1, np.max(y_range) + 0.05)])

        # Plot trajectory comparison
        ax3.plot(t_l_numpy, t_r_numpy, label='target trajectory', linewidth=2, alpha=0.8)
        ax3.plot(o_l_numpy, o_r_numpy, label='predicted trajectory', linewidth=2, alpha=0.8, linestyle='--')
        ax3.set_xlabel('Iris X Position')
        ax3.set_ylabel('Iris Y Position')
        ax3.set_title('Iris Trajectory: Target vs Predicted')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.axis('equal')  # Equal aspect ratio for proper trajectory visualization

        # Add error information to the plot
        error_x = np.abs(t_l_numpy - o_l_numpy)
        error_y = np.abs(t_r_numpy - o_r_numpy)
        mean_error_x = np.mean(error_x)
        mean_error_y = np.mean(error_y)
        fig.suptitle(
            f'Iris Position Tracking - Epoch {epoch}\nMean Error X: {mean_error_x:.4f}, Mean Error Y: {mean_error_y:.4f}',
            fontsize=14, y=0.98)

        # Adjust the spacing between subplots
        plt.subplots_adjust(hspace=0.4, top=0.92)

        picname = f'event_plot_epoch{epoch}.png'
        plt.savefig(os.path.join(plot_dir, picname))
        plt.close()

        # Enhanced frame visualization
        frames_plot_np = np.array(
            frames_plot[0].permute(1, 2, 3, 0).reshape(-1, frames_plot.shape[-2], frames_plot.shape[-1]).cpu())
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))

        for i, ax in enumerate(axs.flatten()):
            if i >= frames_plot_np.shape[0]:
                ax.axis('off')
                continue

            # Plot the image
            ax.imshow(frames_plot_np[i], cmap='gray', alpha=0.8)

            # Plot target position (blue circle)
            target_x, target_y = t_l_numpy[i] * width, t_r_numpy[i] * height
            ax.plot(target_x, target_y, 'bo', markersize=8, markeredgewidth=2, alpha=0.8,
                    label='target' if i == 0 else "")

            # Plot predicted position (red cross)
            pred_x, pred_y = o_l_numpy[i] * width, o_r_numpy[i] * height
            ax.plot(pred_x, pred_y, 'rx', markersize=8, markeredgewidth=2, alpha=0.8,
                    label='predicted' if i == 0 else "")

            # Draw line between target and prediction if they differ significantly
            distance = np.sqrt((target_x - pred_x) ** 2 + (target_y - pred_y) ** 2)
            if distance > 2.0:  # Only draw if error is significant
                ax.plot([target_x, pred_x], [target_y, pred_y], 'g-', alpha=0.6, linewidth=2,
                        label='error' if i == 0 else "")

            # Add frame number
            ax.text(5, 15, f'Frame {i}', color='white', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

            # Hide the axes
            ax.axis('off')

        # Add legend to the figure
        handles, labels = axs[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95),
                       ncol=3, fontsize=12)

        fig.suptitle(f'Eye Tracking Results - Epoch {epoch}\nTarget (blue) vs Predicted (red cross)',
                     fontsize=14, y=0.98)

        picname2 = f'eye_plot_epoch{epoch}.png'
        plt.savefig(os.path.join(plot_dir, picname2), dpi=150, bbox_inches='tight')
        plt.close()

    print("Training completed!")