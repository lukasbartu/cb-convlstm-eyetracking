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

pretrained = True
height = N = 60
width = M = 80
batch_size = 16
seq = 40
stride = 1
stride_val = 40
chunk_size = 500
num_epochs = 20  # More epochs for SNN convergence

# reproducibility
seed = 3
torch.manual_seed(seed)
np.random.seed(seed)

log_dir = f'adam_l1_snn_log_s{seed}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
plot_dir = f'adam_l1_snn_plot_s{seed}'
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
            sample_resize = np.expand_dims(np.array(sample_resize), axis=1)

        label1 = self.target[index][:, 0] / M / (8)
        label2 = self.target[index][:, 1] / N / (8)
        label = np.concatenate([label1.reshape(-1, 1), label2.reshape(-1, 1)], axis=1)

        return torch.from_numpy(sample_resize), torch.from_numpy(label)

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


# ---------------- path helpers ----------------

def load_filenames(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print("ROOT_DIR:", ROOT_DIR)


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


# ---------------- data paths ----------------
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


# ---------------- SNN ConvLSTM Cell ----------------
class SNNConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, beta=0.9, threshold=1.0):
        super(SNNConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.threshold = threshold

        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                                    kernel_size=kernel_size, padding=self.padding, bias=bias)

        spike_grad = surrogate.atan(alpha=2.0)

        self.lif_input = snn.Synaptic(alpha=0.9, beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.lif_forget = snn.Synaptic(alpha=0.9, beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.lif_cell = snn.Synaptic(alpha=0.9, beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.lif_output = snn.Synaptic(alpha=0.9, beta=beta, spike_grad=spike_grad, threshold=threshold)

        self.bn_gates = nn.BatchNorm2d(4 * hidden_dim)

    def forward(self, input_tensor, cur_state=None):
        batch_size, _, height, width = input_tensor.size()

        if cur_state is None:
            h_cur = torch.zeros(batch_size, self.hidden_dim, height, width,
                                device=input_tensor.device, dtype=input_tensor.dtype)
            c_cur = torch.zeros(batch_size, self.hidden_dim, height, width,
                                device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)
        combined_conv = self.bn_gates(combined_conv)  # Normalize for stability

        cc_i, cc_f, cc_g, cc_o = torch.split(combined_conv, self.hidden_dim, dim=1)

        # Simplified SNN approach - use membrane potentials directly for better gradient flow
        i, _, _ = self.lif_input(torch.sigmoid(cc_i))
        f, _, _ = self.lif_forget(torch.sigmoid(cc_f))
        g, _, _ = self.lif_cell(torch.tanh(cc_g))
        o, _, _ = self.lif_output(torch.sigmoid(cc_o))

        # LSTM cell state update
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)

    def reset_states(self):
        """Reset neuron states to prevent gradient accumulation"""
        for neuron in [self.lif_input, self.lif_forget, self.lif_cell, self.lif_output]:
            if hasattr(neuron, 'syn') and neuron.syn is not None:
                neuron.syn = neuron.syn.detach() * 0
            if hasattr(neuron, 'mem') and neuron.mem is not None:
                neuron.mem = neuron.mem.detach() * 0

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_gates.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_gates.weight.device))


class SNNConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1, batch_first=False, bias=True, beta=0.9):
        super(SNNConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            cell_list.append(SNNConvLSTMCell(cur_input_dim, hidden_dim, kernel_size, bias, beta, threshold=1.0))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        batch_size, seq_len, _, height, width = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (height, width))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, (h, c) = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        if not self.batch_first:
            layer_output_list = [l.permute(1, 0, 2, 3, 4) for l in layer_output_list]

        return layer_output_list, last_state_list

    def reset_states(self):
        """Reset all neuron states in all cells"""
        for cell in self.cell_list:
            cell.reset_states()

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


# ---------------- SNN Model matching CNN architecture ----------------
class MySNNModel(nn.Module):
    def __init__(self, height, width, input_dim, beta=0.9):
        super(MySNNModel, self).__init__()

        # Compressed SNN ConvLSTM architecture for reduced model size
        self.convlstm1 = SNNConvLSTM(input_dim=input_dim, hidden_dim=8, kernel_size=(3, 3), num_layers=1,
                                     batch_first=True, beta=beta)
        self.bn1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm2 = SNNConvLSTM(input_dim=8, hidden_dim=16, kernel_size=(3, 3), num_layers=1, batch_first=True,
                                     beta=beta)
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm3 = SNNConvLSTM(input_dim=16, hidden_dim=32, kernel_size=(3, 3), num_layers=1, batch_first=True,
                                     beta=beta)
        self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm4 = SNNConvLSTM(input_dim=32, hidden_dim=64, kernel_size=(3, 3), num_layers=1, batch_first=True,
                                    beta=beta)
        self.bn4 = nn.BatchNorm3d(64)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # Compressed fully connected layers
        self.fc1 = nn.Linear(960, 128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def reset_snn_states(self):
        """Reset all SNN neuron states"""
        self.convlstm1.reset_states()
        self.convlstm2.reset_states()
        self.convlstm3.reset_states()
        self.convlstm4.reset_states()

    def forward(self, x):
        x, _ = self.convlstm1(x)
        x = x[0].permute(0, 2, 1, 3, 4)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = x.permute(0, 2, 1, 3, 4)
        x, _ = self.convlstm2(x)
        x = x[0].permute(0, 2, 1, 3, 4)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.permute(0, 2, 1, 3, 4)
        x, _ = self.convlstm3(x)
        x = x[0].permute(0, 2, 1, 3, 4)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = x.permute(0, 2, 1, 3, 4)
        x, _ = self.convlstm4(x)
        x = x[0].permute(0, 2, 1, 3, 4)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        x_list = []
        b, c, seq, h, w = x.size()
        for t in range(seq):
            data = x[:, :, t, :, :]
            data = data.reshape(b, -1)
            data = F.relu(self.fc1(data))
            data = self.drop(data)
            data = self.fc2(data)
            x_list.append(data)
        y = torch.stack(x_list, dim=0)
        y = y.permute(1, 0, 2)
        return y

input_dim = 1
model = MySNNModel(height, width, input_dim, beta=0.9)
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if pretrained:
    print("Looking for pretrained model....")
    checkpoint_path = 'checkpoint_snn.pth'
    cnn_checkpoint_path = 'checkpoint.pth'  # Original CNN checkpoint

    # Try to load SNN checkpoint first
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"Loaded SNN checkpoint. Loss: {loss:.4f}")
        except Exception as e:
            print(f"Failed to load SNN checkpoint: {e}")
            print("SNN checkpoint architecture doesn't match current model. Starting fresh training...")

    # Try to load CNN checkpoint and transfer compatible weights
    elif os.path.exists(cnn_checkpoint_path):
        print("No SNN checkpoint found. Attempting to transfer weights from CNN checkpoint...")
        try:
            cnn_checkpoint = torch.load(cnn_checkpoint_path)
            cnn_state_dict = cnn_checkpoint['model_state_dict']

            # Transfer compatible weights (FC layers should match exactly)
            model_dict = model.state_dict()
            compatible_weights = {}

            # Transfer fully connected layers
            for key in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']:
                if key in cnn_state_dict and key in model_dict:
                    if cnn_state_dict[key].shape == model_dict[key].shape:
                        compatible_weights[key] = cnn_state_dict[key]
                        print(f"Transferred {key}: {cnn_state_dict[key].shape}")
                    else:
                        print(
                            f"Shape mismatch for {key}: CNN {cnn_state_dict[key].shape} vs SNN {model_dict[key].shape}")

            # Load the compatible weights
            if compatible_weights:
                model_dict.update(compatible_weights)
                model.load_state_dict(model_dict)
                print(f"Successfully transferred {len(compatible_weights)} compatible layers from CNN checkpoint")
            else:
                print("No compatible layers found. Starting fresh training...")

        except Exception as e:
            print(f"Failed to load CNN checkpoint: {e}")
            print("Starting fresh training...")
    else:
        print("No checkpoints found. Starting fresh training...")

model.train()
best_val_loss = float('inf')

for epoch in range(num_epochs):
    running_loss = 0.0
    total_data = len(train_dataloader)
    for t, data in tqdm.tqdm(enumerate(train_dataloader, 0), total=total_data):
        images, targets = data
        images = images.to(device).float()
        targets = targets.to(device).float()

        # Reset SNN states before each batch to prevent gradient accumulation
        model.reset_snn_states()

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        prev_output = outputs
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Validation
    val_running_loss = 0
    num_values = 0
    num_values_3 = 0
    num_values_5 = 0
    num_values_1 = 0
    tot_values = 0
    model.eval()

    with torch.no_grad():
        for images, targets in valid_dataloader:
            images = images.to(device).float()
            targets = targets.to(device).float()

            outputs = model(images)
            val_loss = criterion(outputs, targets)
            dis = targets - outputs
            dis[:, :, 0] *= height
            dis[:, :, 1] *= width
            dist = torch.norm(dis, dim=-1)

            num_values = num_values + torch.sum(dist > 10)
            num_values_5 = num_values_5 + torch.sum(dist > 5)
            num_values_3 = num_values_3 + torch.sum(dist > 3)
            num_values_1 = num_values_1 + torch.sum(dist > 1)
            tot_values = tot_values + dist.numel()
            val_running_loss += val_loss.item()

        val_epoch_loss = val_running_loss / len(valid_dataloader)
        err_rate = num_values / tot_values
        err_rate_3 = num_values_3 / tot_values
        err_rate_5 = num_values_5 / tot_values
        err_rate_1 = num_values_1 / tot_values
        print(f"Validation Loss: {val_epoch_loss:.4f}")
        print(f"err_rate: {err_rate:.4f}")

        # File path
        file_path = os.path.join(log_dir, 'training_log.txt')
        with open(file_path, 'a') as f:
            f.write(
                f"Size {height}, Epoch {epoch}, Loss: {val_epoch_loss}, err_rate_1:{err_rate_1}, err_rate_3:{err_rate_3}, err_rate_5:{err_rate_5}  err: {err_rate} num_values: {num_values} tot_values: {tot_values}\n")

        # Save the model if it has the best validation loss so far
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            print("saving best model...")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, 'checkpoint_snn.pth')

        # Plotting - match CNN version exactly
        valid_dataloader_plt = DataLoader(val_dataset, batch_size=100, shuffle=False)

        for t, data in enumerate(valid_dataloader_plt):
            if t == 1:
                break
            images, targets = data
            frames_plot = images.to(device).float()
            target_plot = targets.to(device).float()
            t_l_list = []
            t_r_list = []
            o_l_list = []
            o_r_list = []

            for i in range(len(frames_plot)):
                images = frames_plot[i].unsqueeze(0)
                outputs = model(images)
                targets = target_plot[i]
                t_l = np.array(targets[:, 0].cpu()).flatten()
                t_r = np.array(targets[:, 1].cpu()).flatten()
                o_l = outputs.detach().cpu().numpy()[:, :, 0].flatten()
                o_r = outputs.detach().cpu().numpy()[:, :, 1].flatten()
                t_l_list.append(t_l)
                t_r_list.append(t_r)
                o_l_list.append(o_l)
                o_r_list.append(o_r)

            t_l_numpy = np.array(t_l_list).flatten()
            t_r_numpy = np.array(t_r_list).flatten()
            o_l_numpy = np.array(o_l_list).flatten()
            o_r_numpy = np.array(o_r_list).flatten()

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
            fig.suptitle(f'Iris Position Tracking - Epoch {epoch}\nMean Error X: {mean_error_x:.4f}, Mean Error Y: {mean_error_y:.4f}',
                        fontsize=14, y=0.98)

            # Adjust the spacing between subplots
            plt.subplots_adjust(hspace=0.4, top=0.92)

            picname = f'event_plot_{epoch}.png'
            plt.savefig(os.path.join(plot_dir, picname))
            plt.close()

            frames_plot = np.array(frames_plot.reshape(-1, frames_plot.shape[-2], frames_plot.shape[-1]).cpu())
            fig, axs = plt.subplots(4, 4, figsize=(12, 12))

            for i, ax in enumerate(axs.flatten()):
                if i < len(frames_plot):
                    # Plot the image
                    ax.imshow(frames_plot[i], cmap='gray', alpha=0.8)

                    # Plot target position (blue circle)
                    target_x, target_y = t_l_numpy[i] * width, t_r_numpy[i] * height
                    ax.plot(target_x, target_y, 'bo', markersize=8, markeredgewidth=2, alpha=0.8,
                           label='target' if i == 0 else "")

                    # Plot predicted position (red cross)
                    pred_x, pred_y = o_l_numpy[i] * width, o_r_numpy[i] * height
                    ax.plot(pred_x, pred_y, 'rx', markersize=8, markeredgewidth=2, alpha=0.8,
                           label='predicted' if i == 0 else "")

                    # Draw line between target and prediction if they differ significantly
                    distance = np.sqrt((target_x - pred_x)**2 + (target_y - pred_y)**2)
                    if distance > 2.0:  # Only draw if error is significant
                        ax.plot([target_x, pred_x], [target_y, pred_y], 'g-', alpha=0.6, linewidth=2,
                               label='error' if i == 0 else "")

                    # Add frame number
                    ax.text(5, 15, f'Frame {i}', color='white', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

                # Hide the axes
                ax.axis('off')

            # Add legend to the figure
            handles, labels = axs[0,0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95),
                          ncol=3, fontsize=12)

            fig.suptitle(f'Eye Tracking Results - Epoch {epoch}\nTarget (blue) vs Predicted (red cross)',
                        fontsize=14, y=0.98)

            picname2 = f'eye_plot_{epoch}.png'
            plt.savefig(os.path.join(plot_dir, picname2), dpi=150, bbox_inches='tight')
            plt.close()

    model.train()  # Switch back to training mode
