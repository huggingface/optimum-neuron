from torch_neuronx.xla_impl.ops import xla_hlo_call


@xla_hlo_call
def fill_prefix(tensor, update):
    scribe = tensor.scribe
    dtype = tensor.dtype
    shape = tensor.sizes
    start_indices = [scribe.u32.Constant(constant_value=0)] * len(shape)
    return dtype[shape].DynamicUpdateSlice(tensor, update, *start_indices)
