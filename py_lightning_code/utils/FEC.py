'''构建FEC选择类，通过type参数进行FEC的编码选择，构建编码器和解码器'''
import tensorflow as tf
import sionna
import math
class Choose_FEC:
    # 构造函数
    def __init__(self, fec_type="LDPC", *args):
        """初始化type类型的编码器和解码器"""
        self.type = fec_type # 默认LDPC
        if self.type == "LDPC":
            if len(args) != 2:
                raise ValueError("LDPC 需要传入 (k, n)")
            self.k , self.n = args
            self.ldpc(self.k , self.n)

    # 1. LDPC
    def ldpc(self, k , n):
        """初始化 LDPC 编码器和解码器"""
        # k = 输入比特数（信息比特长度）
        # n = 输出码字长度（编码后的比特数）
        self.code_rate = k / n # 码率
        self.encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(k, n)
        self.decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)

    # LDPC_CODE
    def ldpc_encoder(self, initial_bit_sequence):
        # 假设 initial_bit_sequence.shape = [BATCH_SIZE, SEQ_LENGTH]
        BATCH_SIZE, SEQ_LENGTH = initial_bit_sequence.shape

        #if SEQ_LENGTH % self.k != 0:
            #raise ValueError("请调整LDPC的k变量！")

        # 计算总码字数量
        total_codewords = math.ceil(SEQ_LENGTH / self.k)
        padded_length = total_codewords * self.k
        pad_length = padded_length - SEQ_LENGTH  # may be 0

        # pad with zeros if needed
        if pad_length != 0:
            padding = tf.zeros([BATCH_SIZE, pad_length], dtype=initial_bit_sequence.dtype)
            bit_seq_padded = tf.concat([initial_bit_sequence, padding], axis=1)
        else:
            bit_seq_padded = initial_bit_sequence

        # reshape to have one codeword per row
        bit_seq_reshaped = tf.reshape(bit_seq_padded, [-1, self.k])  # shape: [BATCH_SIZE * total_codewords, k]

        # 调试：检查输入数据类型和形状

        # encode (LDPC5GEncoder expects last dim == k)
        encoded_flat = self.encoder(bit_seq_reshaped)  # shape: [BATCH_SIZE * total_codewords, n]

        # reshape back to batch form
        encoded_bits = tf.reshape(encoded_flat, [BATCH_SIZE, total_codewords * self.n])

        return encoded_bits, pad_length

    # LDPC_DECODE
    def ldpc_decoder(self, receive_encoded_bits, pad_length):
        """
        encoded_bit_sequence: shape [BATCH_SIZE, num_codewords*n]
        original_bit_length: 可选，原始比特序列长度，用于去掉填充
        """
        BATCH_SIZE, total_bits = receive_encoded_bits.shape

        # 计算总码字数量
        if total_bits % self.n != 0:
            raise ValueError("输入长度不是 n 的整数倍，说明数据有问题")

        num_codewords = total_bits // self.n

        # reshape 成 [BATCH_SIZE*num_codewords, n]
        encoded_bits_reshaped = tf.reshape(receive_encoded_bits, [-1, self.n])

        # 解码
        decoded_bits = self.decoder(encoded_bits_reshaped)

        # reshape 回 [BATCH_SIZE, num_codewords*k]
        decoded_bits = tf.reshape(decoded_bits, [BATCH_SIZE, num_codewords*self.k])

        original_bit_length = decoded_bits.shape[1] - pad_length
        # if original_bit_length provided, trim the padded bits
        if original_bit_length != decoded_bits.shape[1]:
            if original_bit_length > decoded_bits.shape[1]:
                raise ValueError("original_bit_length greater than decoded length")
            decoded_bits_trimmed = decoded_bits[:, :original_bit_length]
            return decoded_bits_trimmed
        else:
            return decoded_bits

        return decoded_bits