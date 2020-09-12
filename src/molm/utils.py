import logging


def get_logger(name, level=logging.INFO):
    """
    :param name: name of the logger
    :param level: level of the logger. 
        logging.DEBUG = 10
               .INFO = 20
               .WARNING = 30
               .CRITICAL = 40
               .CRITICAL = 50
    """
    logger = logging.Logger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def get_n_params(Network):
    """
    :param Network: instance of torch.nn.Module
    :returns: the number of trainable parameters in the Module
    """
    n_params = 0
    sizes = {}
    for param_tensor in Network.state_dict():
        sizes[param_tensor] = Network.state_dict()[param_tensor].size()
        n_params += Network.state_dict()[param_tensor].numel()
    return n_params, sizes
