from .. import context


POSTFIX_MAPPING: dict[str, str] = {
    'UNKNOWN' : '',
    'CAST_RINT' : 'r',
    'CAST_ROUND' : 'a',
    'CAST_FLOOR' : 'f',
    'CAST_CEIL' : 'c',
    'CAST_TRUNC' : 'z',
}


class RoundModeInst:
    def __init__(self, mode: str):
        self.mode = mode 
    
    def __str__(self):
        if context.device_type.startswith('910b') and self.mode=='UNKNOWN':
            return 'CAST_NONE'
        return self.mode
    
    def __eq__(self, other: 'RoundModeInst'):
        return self.mode==other.mode

    @property
    def postfix(self):
        return POSTFIX_MAPPING[self.mode]


class RoundMode:
    NONE = RoundModeInst('UNKNOWN')
    TO_EVEN = RoundModeInst('CAST_RINT')              # R
    AWAY_FROM_ZERO = RoundModeInst('CAST_ROUND')      # A
    FLOOR = RoundModeInst('CAST_FLOOR')               # F
    CEIL = RoundModeInst('CAST_CEIL')                 # C
    TRUNC = RoundModeInst('CAST_TRUNC')               # Z
