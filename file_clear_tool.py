import os

[os.remove(os.path.join('content/chip/images/train', _file)) for _file in os.listdir('content/chip/images/train') if _file.endswith('.tif')]
[os.remove(os.path.join('content/chip/images/valid', _file)) for _file in os.listdir('content/chip/images/valid') if _file.endswith('.tif')]
[os.remove(os.path.join('content/chip/labels/train', _file)) for _file in os.listdir('content/chip/labels/train') if _file.endswith('.txt')]
[os.remove(os.path.join('content/chip/labels/valid', _file)) for _file in os.listdir('content/chip/labels/valid') if _file.endswith('.txt')]
