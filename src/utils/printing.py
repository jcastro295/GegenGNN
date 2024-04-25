"""

``printing.py``
----------------

Basic function to format text to be shown in console.
colors_dict contains supported colors.
styles_dict contains available styles.

"""

colors_dict = {
    'default' : '\033[99m',
    'purple' : '\033[95m',
    'grey': '\033[30m',
    'black': '\033[90m',
    'cyan' : '\033[96m',
    'darkcyan' : '\033[36m',
    'blue' : '\033[94m',
    'green' : '\033[92m',
    'yellow' : '\033[93m',
    'red' : '\033[91m',
    'magenta': '\033[95m',
    'white': '\033[97m'
}

styles_dict = {
    'default' : '\033[99m',
    'bold' : '\033[1m',
    'underline' : '\033[4m',
    'italic' : '\033[3m',
    'underline_bold' : '\033[4m\033[1m',
    'underline_italic' : '\033[4m\033[3m',
    'italic_bold' : '\033[3m\033[1m',
    'all' : '\033[4m\033[3m\033[1m'
}


def color_text(text, color='default', style='default'):
    """
    Turn text to a different color and format to display in console

    Parameters
    ----------
    text : `str`
        Text to be converted
    color : `str, default='default'`
        String with colors to print. List of available colors:
        `'default'`, `'purple'`, `'grey'`, `'black'`, `'cyan'`, `'darkcyan'`, `'blue'`,
        `'green'`, `'yellow'`, `'red'`, `'magenta'`, `'white'`
    style : `str, defatul=None`
        String with style to be applied. List of available styles:
        `'bold`, `'underline'`, `'italic'`, `'underline_bold'`, `'underline_italic'`,
        `'italic_bold'` and `'all'`

    Returns
    -------
    `str`
        Converted text
    """
    color = colors_dict.get(color, '\033[99m')
    style = styles_dict.get(style, '\033[99m')

    return f"{style}{color}{text}\033[0m"
