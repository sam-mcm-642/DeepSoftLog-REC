
class DummyAutocast:
    def __init__(self, enabled=True):
        self.enabled = enabled
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class DummyAmp:
    def __init__(self):
        self.autocast = DummyAutocast
    
    def initialize(self, models, optimizers=None, enabled=True, opt_level="O1", **kwargs):
        return models, optimizers
    
    def float_function(self, func):
        return func

amp = DummyAmp()
