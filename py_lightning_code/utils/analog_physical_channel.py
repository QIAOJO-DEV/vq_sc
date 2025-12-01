import torch
import torch.nn as nn
import numpy as np


class analog_channel(nn.Module):
    """
    Currently the channel model is either error free,
    rayleigh channel or the AWGN channel.
    """

    def __init__(self, chan_type='awgn'):
        super(analog_channel, self).__init__()
        self.chan_type = chan_type


    # -------------------------
    # Noise layers
    # -------------------------
    def gaussian_noise_layer(self, input_layer, std):
        device = input_layer.device
        noise_real = torch.normal(0.0, std, size=input_layer.shape, device=device)
        noise_imag = torch.normal(0.0, std, size=input_layer.shape, device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std):
        device = input_layer.device
        noise_real = torch.normal(0.0, std, size=input_layer.shape, device=device)
        noise_imag = torch.normal(0.0, std, size=input_layer.shape, device=device)
        noise = noise_real + 1j * noise_imag

        h = torch.sqrt(
            torch.normal(0.0, 1.0, size=input_layer.shape, device=device) ** 2 +
            torch.normal(0.0, 1.0, size=input_layer.shape, device=device) ** 2
        ) / np.sqrt(2)

        return (input_layer * h + noise)/h

    # -------------------------
    # Normalize power
    # -------------------------
    def complex_normalize(self, x, power=1):
        pwr = torch.mean(x ** 2) * 2
        out = np.sqrt(power) * x / torch.sqrt(pwr)
        return out, pwr

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, input, chan_param, avg_pwr=False):
        device = input.device

        # Normalize
        if avg_pwr:
            channel_tx = input / torch.sqrt(avg_pwr * 2)
            pwr = avg_pwr
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1)

        # reshape -> complex
        input_shape = channel_tx.shape
        flat = channel_tx.reshape(-1)
        L = flat.shape[0]

        channel_in = flat[:L // 2] + 1j * flat[L // 2:]

        # channel
        channel_output = self.complex_forward(channel_in, chan_param)

        # complex -> real/imag concat
        channel_output = torch.cat(
            [torch.real(channel_output), torch.imag(channel_output)]
        )
        channel_output = channel_output.reshape(input_shape)

        # AWGN
        if self.chan_type == 'awgn' or self.chan_type == 1:
            noise = (channel_output - channel_tx).detach()
            noise.requires_grad = False
            out = channel_tx + noise
            return out * torch.sqrt(pwr)

        # Rayleigh
        elif self.chan_type == 'rayleigh' or self.chan_type == 2:
            return channel_output * torch.sqrt(pwr)

        else:
            return input

    # -------------------------
    # complex_forward
    # -------------------------
    def complex_forward(self, channel_in, chan_param):
        if self.chan_type == 'none' or self.chan_type == 0:
            return channel_in

        sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))

        if self.chan_type == 'awgn' or self.chan_type == 1:
            return self.gaussian_noise_layer(channel_in, sigma)

        elif self.chan_type == 'rayleigh' or self.chan_type == 2:
            return self.rayleigh_noise_layer(channel_in, sigma)

        return channel_in


# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    channel = analog_channel(chan_type='awgn')
    x = torch.randn(4, 16).to("cuda")  # ANY tensor on ANY device
    y = channel(x, chan_param=10)
    print(y.shape)