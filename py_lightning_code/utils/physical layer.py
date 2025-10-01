import torch
from torchvision import transforms
from PIL import Image
import tensorflow as tf
from FEC import Choose_FEC
from Modulation import Modulation
from to_patch import spilit2patch
from my_utils import split2bitstream
import sionna
class PhysicalLayer:
    """
    物理层传输

    """
    def __init__(self, num_bits_per_symbol,modulation_type="qam",fec_type="LDPC",crc_type="CRC24A",ebno_db= 10,channel_type="awgn"):
        k=64
        n=128
        self.channel_type=channel_type
        self.fec=Choose_FEC(fec_type,k,n)
        self.num_bits_per_symbol=num_bits_per_symbol
        self.modulation=Modulation(modulation_type,num_bits_per_symbol)
        self.crc_encoder=sionna.phy.fec.crc.CRCEncoder(crc_degree=crc_type)
        self.crc_decoder=sionna.phy.fec.crc.CRCDecoder(crc_encoder=self.crc_encoder)
        self.ebno_db = ebno_db
    def pass_channel(self,bitstream):
        """
        物理层传输：
        1. CRC编码
        2. FEC编码
        3. 调制
        """
        bitstream=tf.cast(bitstream,tf.float32)
        bitstream=self.crc_encoder(bitstream)
        bitstream, pad_code = self.fec.ldpc_encoder(bitstream)
        modulated_symbols,pad_mod = self.modulation.modulation(bitstream)
        if self.channel_type == "awgn":
          my_channel = sionna.phy.channel.AWGN()
          no = sionna.phy.utils.ebnodb2no(self.ebno_db, self.num_bits_per_symbol, coderate=1)
        modulated_symbols=my_channel(modulated_symbols,no)
        rec_bitstream=self.modulation.demodulation(modulated_symbols,no,pad_mod)
        rec_bitstream=self.fec.ldpc_decoder(rec_bitstream,pad_code)
        rec_bitstream,valid=self.crc_decoder(rec_bitstream)
        return rec_bitstream,valid


    def bit2symbol(self,modulated_symbols):
        """
        """
    def symbol2bit(self,received_symbols):
        """
        物理层接收：
        1. CRC解码
        2. FEC解码
        3. 解调
        """
if __name__ == "__main__":
    x = torch.randint(0,2,(33,1),dtype=torch.int64)
    print(x)
    split_bit=split2bitstream(9,x.shape,x.dtype)
    x=split_bit.tensor_to_bits(x)
    split_patch=spilit2patch(x.shape,x.dtype)
    x=split_patch.tensor_to_patch(x)
    physical_layer=PhysicalLayer(num_bits_per_symbol=4)
    rec_bitstream,valid=physical_layer.pass_channel(x)
    rec_bitstream=split_patch.patch_to_tensor(rec_bitstream)
    rec_bitstream=split_bit.bits_to_tensor(rec_bitstream)
    print(rec_bitstream)
    print(valid)
