import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, Tuple
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
import math
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
from brevitas.core.scaling import ScalingImplType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.quant.base import QuantType
from torch.utils.cpp_extension import load

class AdaPT_Linear_Function_Brevitas(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, bias_, input_quant, weight_quant, max_value, axx_linear_kernel):
        ctx.save_for_backward(input, weight, bias)
        ctx.bias_ = bias_
        
        # Quantize with Brevitas
        quant_weight, weight_scale, _ = weight_quant(weight)
        quant_input, input_scale, _ = input_quant(input)
        
        # Convert to int8
        quant_input = quant_input.to(dtype=torch.int8)
        quant_weight = quant_weight.to(dtype=torch.int8)
        
        # Scaling logic
        output = axx_linear_kernel.forward(quant_input, quant_weight)
        output = output / ((max_value/input_scale)*(max_value/weight_scale))
                                      
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


class AdaPT_Linear_Brevitas(nn.Module):
    def __init__(self, size_in, size_out, bias=True, axx_mult='mul8s_acc'):
        super(AdaPT_Linear_Brevitas, self).__init__()
        
        self.size_in, self.size_out, self.bias_ = size_in, size_out, bias
        self.fn = AdaPT_Linear_Function_Brevitas.apply
        
        # weight/bias initialization
        weight = torch.Tensor(size_out, size_in)
        self.weight = nn.Parameter(weight)
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)
        self.axx_mult = axx_mult
        
        # Initialization logic
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
               
        # Original quantization parameters
        num_bits=8
        unsigned=False
        self.max_value = pow(2,num_bits-1)-1  # 127 for signed 8-bit

        # Brevitas quantizers replacing QuantDescriptor/TensorQuantizer
        self.input_quant = Int8ActPerTensorFloat(
            scaling_impl_type=ScalingImplType.PARAMETER,
            restrict_scaling_type=RestrictValueType.LOG_FP,
            quant_type=QuantType.INT,
            bit_width_impl_type=None,
            narrow_range=True,
            float_to_int_impl_type=None)
        
        self.weight_quant = Int8WeightPerTensorFloat(
            scaling_impl_type=ScalingImplType.PARAMETER,
            restrict_scaling_type=RestrictValueType.LOG_FP,
            quant_type=QuantType.INT,
            bit_width_impl_type=None,
            narrow_range=True,
            float_to_int_impl_type=None)
        
        # Original kernel loading
        self.axx_linear_kernel = load(
            name='PyInit_linear_'+axx_mult, 
            sources=["/scratch-local/khan/low_bit_quantization/cds_bre_quant/adapt/adapt/cpu-kernels/axx_linear.cpp"], 
            extra_cflags=['-DAXX_MULT=' + axx_mult + ' -march=native -fopenmp -O3'], 
            extra_ldflags=['-lgomp'], 
            verbose=True)
       
    def forward(self, x):       
        x = self.fn(x, self.weight, self.bias, self.bias_, 
                   self.input_quant, self.weight_quant, 
                   self.max_value, self.axx_linear_kernel)
        return x


class AdaPT_Conv2d_Function_Brevitas(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, input_quant, weight_quant, kernel_size, 
                max_value, out_channels, bias_, axx_conv2d_kernel, 
                bias=None, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
                         
        if padding_mode != 'zeros':
            return F.conv2d(F.pad(input, ctx._reversed_padding_repeated_twice, mode=padding_mode),
                            weight, bias, stride,
                            _pair(0), dilation, groups)
                    
        # Quantize with Brevitas
        quant_weight, weight_scale, _ = weight_quant(weight)
        quant_input, input_scale, _ = input_quant(input)
        
        # Original int8 conversion
        quant_input = quant_input.to(dtype=torch.int8)
        quant_weight = quant_weight.to(dtype=torch.int8)
                        
        if groups > 1:
            out=torch.empty(0)
            for i in range(0,groups):
                filters = quant_weight[i:(i+1)]                   
                o = axx_conv2d_kernel.forward(quant_input[:, i:(i+1)], filters, kernel_size, stride, padding) 
                out = torch.cat((out, o), dim=1)
            
            out = out/((max_value/input_scale)*((max_value/weight_scale)))
            if bias_:
                return out + bias.reshape(1,out_channels,1,1)   
            else: 
                return out
        
        out = axx_conv2d_kernel.forward(quant_input, quant_weight, kernel_size, stride, padding)
        out = out/((max_value/input_scale)*((max_value/weight_scale)))
        
        if bias_:
            return out + bias.reshape(1,out_channels,1,1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None            

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None, None, None


class AdaPT_Conv2d_Brevitas(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        axx_mult='mul8s_acc',
        device=None,
        dtype=None):
        
        super(AdaPT_Conv2d_Brevitas, self).__init__()
        
        # Original parameter handling
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        self.bias_ = bias
        self.axx_mult = axx_mult
        
        # Original weight/bias initialization
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size_))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Original initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Original quantization parameters
        num_bits=8
        unsigned=False
        self.max_value = pow(2,num_bits-1)-1

        # Brevitas quantizers replacing original
        self.input_quant = Int8ActPerTensorFloat(
            scaling_impl_type=ScalingImplType.PARAMETER,
            restrict_scaling_type=RestrictValueType.LOG_FP,
            quant_type=QuantType.INT,
            bit_width_impl_type=None,
            narrow_range=True,
            float_to_int_impl_type=None)
        
        self.weight_quant = Int8WeightPerTensorFloat(
            scaling_impl_type=ScalingImplType.PARAMETER,
            restrict_scaling_type=RestrictValueType.LOG_FP,
            quant_type=QuantType.INT,
            bit_width_impl_type=None,
            narrow_range=True,
            float_to_int_impl_type=None)

        # Original kernel loading
        self.axx_conv2d_kernel = load(
            name='PyInit_conv2d_'+axx_mult, 
            sources=["/scratch-local/khan/low_bit_quantization/cds_bre_quant/adapt/adapt/cpu-kernels/axx_conv2d.cpp"], 
            extra_cflags=['-DAXX_MULT=' + axx_mult + ' -march=native -fopenmp -O3'], 
            extra_ldflags=['-lgomp'], 
            verbose=True)

        # Store other parameters as original
        self.kernel_size = kernel_size_
        self.stride = stride_
        self.padding = padding_
        self.dilation = dilation_
        self.groups = groups
        self.padding_mode = padding_mode
        self.out_channels = out_channels

    def forward(self, input: Tensor) -> Tensor:
        return AdaPT_Conv2d_Function_Brevitas.apply(
            input, self.weight, self.input_quant, self.weight_quant, 
            self.kernel_size, self.max_value, self.out_channels, self.bias_,
            self.axx_conv2d_kernel, self.bias, self.stride, self.padding, 
            self.dilation, self.groups, self.padding_mode)