'''构建Modulation选择类，构建调制器和解调器'''
import tensorflow as tf
import sionna
import math
import matplotlib.pyplot as plt
from sionna.phy.mapping import Constellation
from sionna.phy.mapping import Mapper,Demapper
class Modulation:
    def __init__(self, modulation_type = "qam", num_bits_per_symbol = 2):
        self.modulation_type = modulation_type # 调制类型 默认"qam"
        self.num_bits_per_symbol = num_bits_per_symbol # 默认2 即QPSK
        # 符号映射表（bit → 复数符号）
        self.constellation = Constellation(self.modulation_type, self.num_bits_per_symbol)

    # 展示无数据的星座图
    def base_show(self):
        self.constellation.show()

    # 根据输入的复数展示星座图
    def modulation_show(self, modulation_complex):
        # 将接收到的噪声样本可视化
        plt.figure(figsize=(8,8))
        plt.gca().set_aspect(1)  # 更标准
        plt.grid(True)
        plt.title('Channel output')
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.scatter(tf.math.real(modulation_complex), tf.math.imag(modulation_complex))
        plt.tight_layout()

    # 调制器，将比特映射到星座点。该映射器将星座图作为参数
    def modulation(self, encoder_bit_sequence):
        mapper = Mapper(constellation=self.constellation)

        BATCH_SIZE, SEQ_LENGTH = encoder_bit_sequence.shape
        # 计算总码字数量
        total_codewords = math.ceil(SEQ_LENGTH / self.num_bits_per_symbol)
        padded_length = total_codewords * self.num_bits_per_symbol
        pad_length = padded_length - SEQ_LENGTH  # may be 0

        # pad with zeros if needed
        if pad_length != 0:
            padding = tf.zeros([BATCH_SIZE, pad_length], dtype=encoder_bit_sequence.dtype)
            bit_seq_padded = tf.concat([encoder_bit_sequence, padding], axis=1)
        else:
            bit_seq_padded = encoder_bit_sequence

        # reshape to have one codeword per row
        complex_seq_reshaped = tf.reshape(bit_seq_padded, [-1, self.num_bits_per_symbol])  # shape: [BATCH_SIZE * total_codewords, k]

        # encode (LDPC5GEncoder expects last dim == k)
        modulation_complex = mapper(complex_seq_reshaped)  # shape: [BATCH_SIZE * total_codewords, n]

        # reshape back to batch form
        modulation_complex = tf.reshape(modulation_complex, [BATCH_SIZE, total_codewords])

        return modulation_complex, pad_length

     # 解调器，把接收到的符号映射回比特序列（或者生成软比特/LLR） 这里 "app" 表示 应用层解映射（a posteriori probability），也就是生成软输出
    def demodulation(self, receive_signal_complex, no, pad_length):

        demapper = Demapper("app", constellation=self.constellation)
        llr = demapper(receive_signal_complex, no)
        # 如果有 padding，就去掉尾部多余的 LLR
        if pad_length > 0:
            llr = llr[:, :-pad_length]
        return llr

    # 打印调制后（bit → 复数符号）
    ## num_samples为打印bit数，默认8个
    ## batch_index第几个batch样本，默认第一个;
    ## is_true为是否全打印，默认否
    def print_modulation_complex(self, is_true = False, num_samples = 8, batch_index = 0):
        if is_true:
            print("The entire modulation Complex:",self.modulation_complex)
        else:
             num_symbols = int(num_samples/self.num_bits_per_symbol)  # 对应符号数，打印bit数/每个符号的比特数
             if batch_index < 0 or batch_index >= self.modulation_complex.shape[0]:
                 raise ValueError("打印的batch越界了！")
             print(f"第{batch_index}个batch的前{num_symbols}符号为: {np.round(self.modulation_complex[batch_index,:num_symbols], 2)}") # 打印对应的星座点（调制后的复数），保留2位小鼠

    # 打印解调后的llr （复数符号 → bit）
    ## num_samples为打印bit数，默认8个
    ## batch_index第几个batch样本，默认第一个;
    ## is_true为是否全打印，默认否
    def print_demodulation_llr(self, is_true = False, num_samples = 8, batch_index = 0):
        if is_true:
            print("The entire demodulation LLR:",self.llr)
        else:
             num_symbols = int(num_samples/self.num_bits_per_symbol)  # 对应符号数，打印bit数/每个符号的比特数
             if batch_index < 0 or batch_index >= self.llr.shape[0]:
                 raise ValueError("打印的batch越界了！")
             print(f"第{batch_index}个batch的前{num_symbols}LLR为: {np.round(self.llr[batch_index,:num_samples], 2)}") # 打印解调器输出的前 8 个 LLR 值 每个 LLR 对应一个比特的置信度，正数 → 更可能是 0，负数 → 更可能是 1