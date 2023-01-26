from typing import List, Union

from transformers.models.led.configuration_led import LEDConfig

class LEDHeterformerConfig(LEDConfig):
    def __init__(
        self, 
        attention_window: List[int] = [16,16,32,32,64,64,128,128,256,256,256,256], 
        attention_dilation: Union[List[int], int] = 1,
        autoregressive: bool = False, 
        attention_mode: str = 'sliding_chunks', 
        **kwargs
    ):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Heterformer
                selfattention, 'sliding_chunks' for another implementation of Heterformer selfattention
        """

        super().__init__(**kwargs)

        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        if isinstance(self.attention_dilation, int):
            self.attention_dilation = [self.attention_dilation] * self.encoder_layers

        assert self.attention_mode in ['tvm', 'sliding_chunks', 'n2', 'sliding_chunks_no_overlap']

        