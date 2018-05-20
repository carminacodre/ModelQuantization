DEFAULT_SIZE = 299

optimizing_for_deployment_trans = ['strip_unused_nodes(type=float, shape="1,'
                                   +str(DEFAULT_SIZE)+','+str(DEFAULT_SIZE)+',3")',
                                   'remove_nodes(op=Identity, op=CheckNumerics)',
                                   'fold_constants(ignore_errors=true)',
                                   'fold_batch_norms',
                                   'fold_old_batch_norms']


fix_missing_kernel_errors_trans = ['strip_unused_nodes(type=float, shape="1,'
                                   +str(DEFAULT_SIZE)+','+str(DEFAULT_SIZE)+',3")',
                                   'fold_constants(ignore_errors=true)',
                                   'fold_batch_norms',
                                   'fold_old_batch_norms']

optimize_quantize_weights_trans = ['strip_unused_nodes(type=float, shape="1,'
                                   +str(DEFAULT_SIZE)+','+str(DEFAULT_SIZE)+',3")',
                                   'fold_constants(ignore_errors=true)',
                                   'fold_batch_norms',
                                   'fold_old_batch_norms',
                                   'quantize_weights']

def get_optimizing_for_deployment_for_shape(shape):
    return ['strip_unused_nodes(type=float, shape="1,'
            + str(shape) + ',' + str(shape) + ',3")',
            'remove_nodes(op=Identity, op=CheckNumerics)',
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms',
            'fold_old_batch_norms']

def get_fix_missing_kernel_errors_for_shape(shape):
    return ['strip_unused_nodes(type=float, shape="1,'
            +str(shape)+','+str(shape)+',3")',
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms',
            'fold_old_batch_norms']

def get_optimize_quantize_weights_for_shape(shape):
    return ['strip_unused_nodes(type=float, shape="1,'
            +str(shape)+','+str(shape)+',3")',
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms',
            'fold_old_batch_norms',
            'quantize_weights']
