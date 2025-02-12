import torch
import torch_neuronx
import torch_xla.core.xla_model as xm

'''
    Example with direct XLA call.
    
    Let us take a simple example and see how a simple task might get complex
    with torch xla lowering. 
    
    Consider, a simple module that slices fixed size from 1d tensor starting at 
    any valid index. 
    
    There are many obvious ways to do this on torch. 
    
    tensor = torch.tensor([0,1,...,98,99])
    start = torch.tensor(5)
    
    Say, we want to slice the first 10 values along dim 1 starting from index 5, 

    # 1
    torch.narrow(tensor, dim=1,start=start,length=10)

    # 2
    tensor[:,start:start+10]
    
    # 3
    indices = torch.arange(start, start + size)
    return torch.index_select(tensor, dim=1, index=indices)
    
    
    But there is a problem with here when running on NxD, if you were to trace
    these functions, xla inlines the start position as a constant in the HLO. 
    
    There are two problems with this, 
    a. In JIT flow you will recompile the graph each time you change the start 
       position. 
    b. In Trace(AOT) flow, you will get a compiled model that always slices at 
       the start position passed when tracing. 
    
    Scroll below to see what each of these function lower into. 
    
    Now, if you want the start position to be a parameter a good way to do this
    is to plug into XLA builder. We can use XLA's DynamicSlice operation. 
'''
    
    
def test():
    
    tensor = torch.arange(100, dtype=torch.float32)
    start_pos = torch.tensor(5)
    
    _narrow = torch_neuronx.trace(narrow, example_inputs=(tensor, start_pos), compiler_workdir="/tmp/narrow/")
    _index = torch_neuronx.trace(index_slice, example_inputs=(tensor, start_pos), compiler_workdir="/tmp/index/")
    _index_select = torch_neuronx.trace(index_select, example_inputs=(tensor, start_pos), compiler_workdir="/tmp/index_select/")
    _gather = torch_neuronx.trace(gather, example_inputs=(tensor, start_pos), compiler_workdir="/tmp/gather/")
    _dynamic_slice = torch_neuronx.trace(dynamic_slice, example_inputs=(tensor, start_pos), compiler_workdir="/tmp/dynamic_slice/")

    target = torch.arange(10, 20, dtype=torch.float32)
    assert not torch.allclose(_narrow(tensor, torch.tensor(10)), target)
    assert not torch.allclose(_index(tensor, torch.tensor(10)), target)
    assert not torch.allclose(_index_select(tensor, torch.tensor(10)), target)
    assert not torch.allclose(_gather(tensor, torch.tensor(10)), target)
    
    assert torch.allclose(_dynamic_slice(tensor, torch.tensor(10)), target)
    

def dynamic_slice(tensor, 
                  start_indices):

    # Import annoation from torch_neuronx
    from torch_neuronx.xla_impl.ops import xla_call
    from torch_xla.core import xla_builder as xb        
    
    @xla_call
    def xla_dynamic_slice(tensor: xb.Op, 
                          *start_indices):
        return tensor.dynamic_slice(start_indices, [10])

    return xla_dynamic_slice(tensor, start_indices)


# Case 1: 

def narrow(tensor, start_index):
    return torch.narrow(tensor, dim=0, start=start_index, length=10)

def index_slice(tensor, 
                start_index):
    return tensor[start_index : start_index + 10]

def index_select(tensor, 
                 start_index):
    indices = torch.arange(start_index, start_index + 10, device=tensor.device)
    return torch.index_select(tensor, dim=0, index=indices)

def gather(tensor, 
           start_index):
    indices = torch.arange(start_index, start_index + 10, device=tensor.device)
    return torch.gather(tensor, dim=0, index=indices)

if __name__ == "__main__":
    test()