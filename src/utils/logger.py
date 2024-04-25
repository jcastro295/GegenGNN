"""

`set_logger.py`
---------------

This script intends to set a logger object with a specific name

"""

import logging


def set_logger(name):
    """
    Sets a logger with the module's name

    Parameters
    ----------
    name : `str`
        String with module's name

    Returns
    -------
    `Obj` ``logger``

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    #c_format = logging.Formatter('%(levelname)s\t%(asctime)s\t%(name)s: %(message)s',
    #                             "%Y-%m-%d %H:%M:%S")
    c_format = logging.Formatter(
                        '%(levelname)s\t%(asctime)s\t%(module)s.%(funcName)s() : %(message)s',
                        "%Y-%m-%d %H:%M:%S")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    return logger
