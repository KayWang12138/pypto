class ReduceModeInst():
    def __init__(self, mode: str):
        self.mode = mode 
    
    def __str__(self):
        return self.mode 
    
    def __eq__(self, other: 'ReduceModeInst'):
        if self.mode==other.mode:
            return True 
        else:
            return False 


class ReduceMode():
    ONLY_VALUE = ReduceModeInst('ONLY_VALUE')
    ONLY_INDEX = ReduceModeInst('ONLY_INDEX')
    VALUE_INDEX = ReduceModeInst('VALUE_INDEX')
    INDEX_VALUE = ReduceModeInst('INDEX_VALUE')
