# axx_layers.py
# for quantization
import torch.utils.data
import pytorch_quantization.utils
import pytorch_quantization.nn.modules._utils as _utils
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
import pytorch_quantization.nn as quant_nn

from .torch_utils import _ConvNd, _size_2_t, Union, Tensor, Optional, _pair
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers
import math

from torch.utils.cpp_extension import load

class AdaPT_Linear_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, bias_, quantizer, quantizer_w, max_value, axx_linear_kernel):
        ctx.save_for_backward(input, weight, bias)
        ctx.bias_ = bias_

        quant_input = quantizer(input)
        quant_weight = quantizer_w(weight)

        quant_input_int = quant_input.to(dtype=torch.int8)
        quant_weight_int = quant_weight.to(dtype=torch.int8)

        amax = quantizer.amax if quantizer.amax is not None else torch.tensor(1.0, device=input.device)
        amax_w = quantizer_w.amax if quantizer_w.amax is not None else torch.tensor(1.0, device=input.device)

        output = axx_linear_kernel.forward(quant_input_int, quant_weight_int)
        output = output / ((max_value / amax) * (max_value / amax_w))

        if bias_:
            return output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        bias_ = ctx.bias_

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias_ and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class AdaPT_Linear(nn.Module):
    def __init__(self, size_in, size_out, bias=True, axx_mult='mul8s_acc'):
        super(AdaPT_Linear, self).__init__()
        self.size_in, self.size_out, self.bias_ = size_in, size_out, bias
        self.fn = AdaPT_Linear_Function.apply
        self.weight = nn.Parameter(torch.Tensor(size_out, size_in))
        self.bias = nn.Parameter(torch.Tensor(size_out))
        self.axx_mult = axx_mult

        self.axx_linear_kernel = load(
            name='PyInit_linear_' + axx_mult,
            sources=["//scratch-local/khan/cnn_model/adapt/adapt/cpu-kernels/axx_linear.cpp"],
            extra_cflags=['-DAXX_MULT=' + axx_mult + ' -march=native -fopenmp -O3'],
            extra_ldflags=['-lgomp'],
            verbose=True
        )

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        num_bits = 8
        unsigned = False
        self.max_value = pow(2, num_bits - 1) - 1 if not unsigned else pow(2, num_bits) - 1

        self.quant_desc = QuantDescriptor(num_bits=num_bits, fake_quant=True, unsigned=unsigned, calib_method='max')
        self.quantizer = TensorQuantizer(self.quant_desc)
        self.quantizer_w = TensorQuantizer(self.quant_desc)

        # Disable calibration collection for QAT
        self.quantizer.disable_calib()
        self.quantizer_w.disable_calib()

    def forward(self, x):
        return self.fn(x, self.weight, self.bias, self.bias_, self.quantizer, self.quantizer_w, self.max_value, self.axx_linear_kernel)

class AdaPT_Conv2d_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, quantizer, quantizer_w, kernel_size, max_value, out_channels, bias_, axx_conv2d_kernel, bias=None, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        if padding_mode != 'zeros':
            return F.conv2d(F.pad(input, _pair(padding), mode=padding_mode), weight, bias, stride, _pair(0), dilation, groups)

        quant_input = quantizer(input)
        quant_weight = quantizer_w(weight)
        # breakpoint()
        quant_input_int = quant_input.to(dtype=torch.int8)
        quant_weight_int = quant_weight.to(dtype=torch.int8)
        # breakpoint()
        amax = quantizer.amax if quantizer.amax is not None else torch.tensor(1.0, device=input.device)
        amax_w = quantizer_w.amax if quantizer_w.amax is not None else torch.tensor(1.0, device=input.device)
        print("input shape:", input.shape)
        print("weight shape:", weight.shape)
        print("quant_input_int shape:", quant_input_int.shape)
        print("quant_weight_int shape:", quant_weight_int.shape)

        # breakpoint()
        if groups > 1:
            out = torch.empty(0)
            for i in range(groups):
                filters = quant_weight_int[i:(i+1)]
                o = axx_conv2d_kernel.forward(quant_input_int[:, i:(i+1)], filters, kernel_size, stride, padding)
                out = torch.cat((out, o), dim=1)
            out = out / ((max_value / amax) * (max_value / amax_w))
            if bias_:
                return out + bias.reshape(1, out_channels, 1, 1)
            return out

        out = axx_conv2d_kernel.forward(quant_input_int, quant_weight_int, kernel_size, stride, padding)
        out = out / ((max_value / amax) * (max_value / amax_w))
        # breakpoint()
        if bias_:
            return out + bias.reshape(1, out_channels, 1, 1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None, None

class AdaPT_Conv2d(_ConvNd):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', axx_mult='mul8s_acc', device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        self.bias_ = bias
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        self.axx_mult = axx_mult

        super(AdaPT_Conv2d, self).__init__(in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
                                           False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

        self.axx_conv2d_kernel = load(
            name='PyInit_conv2d_' + axx_mult,
            sources=["//scratch-local/khan/cnn_model/adapt/adapt/cpu-kernels/axx_conv2d.cpp"],
            extra_cflags=['-DAXX_MULT=' + axx_mult + ' -march=native -fopenmp -O3'],
            extra_ldflags=['-lgomp'],
            verbose=True
        )

        if groups != 1 and groups != in_channels:
            raise ValueError('AdaPT_Conv2d does not support groups != in_channels')

        num_bits = 8
        unsigned = False
        self.max_value = pow(2, num_bits - 1) - 1 if not unsigned else pow(2, num_bits) - 1

        self.quant_desc = QuantDescriptor(num_bits=num_bits, fake_quant=True, unsigned=unsigned, calib_method='max')
        self.quantizer = TensorQuantizer(self.quant_desc)
        self.quantizer_w = TensorQuantizer(self.quant_desc)

        self.quantizer.disable_calib()
        self.quantizer_w.disable_calib()

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        return AdaPT_Conv2d_Function.apply(input, weight, self.quantizer, self.quantizer_w, self.kernel_size, self.max_value, self.out_channels, self.bias_, self.axx_conv2d_kernel, bias, self.stride, self.padding, self.dilation, self.groups, self.padding_mode)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
