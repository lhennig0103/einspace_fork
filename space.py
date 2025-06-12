from einspace.search_spaces import EinSpace
from config import my_cfg

class CustomEinSpace(EinSpace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_options = my_cfg
