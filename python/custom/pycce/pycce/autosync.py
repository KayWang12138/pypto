from .utils import Instruction, PIPE, PipeInst
from typing import Literal
from . import context 


class auto_sync():
    def __init__(self) -> None:
        ...
    
    def __enter__(self):
        g_vec = context.active_vec
        g_cube = context.active_cube

        if g_cube is not None:
            g_cube.append(Instruction('STARTAUTOSYNC'))
        elif g_vec is not None:
            g_vec.append(Instruction('STARTAUTOSYNC'))
        else:
            raise Exception('Autosync must be called in VecModule or CubeModule')

    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type:
            # raise exec_type(exec_val)
            return False
        g_vec = context.active_vec
        g_cube = context.active_cube

        if g_cube is not None:
            g_cube.append(Instruction('ENDAUTOSYNC'))
            return True
        elif g_vec is not None:
            g_vec.append(Instruction('ENDAUTOSYNC'))
            return True
        else:
            raise Exception('Autosync must be called in VecModule or CubeModule ')
        
    def __call__(self, fn):
        def wrapped_nograd_fn(*args):
            with auto_sync():
                fn(*args)
        return wrapped_nograd_fn


def parse_autosync(inst_list: list[Instruction], mode: Literal['cube', 'vec']):
    parsed_list: list[Instruction] = []
    tmp_list: list[Instruction] = []

    target_list: list[Instruction] = parsed_list

    for idx, i in enumerate(inst_list):
        if i._inst=='STARTAUTOSYNC':
            target_list = tmp_list
        elif i._inst=='ENDAUTOSYNC':
            res = parse_inner(tmp_list, mode)
            parsed_list += res 
            tmp_list = []
            target_list = parsed_list
        else:
            target_list.append(i)
    return parsed_list
        

PIPE_MAPPING = {
    # mte2
    'GM2UB'                 : PIPE.MTE2,
    # V
    'UB2UB'                 : PIPE.V,
    'UB2UB_ND2NZ'           : PIPE.V,
    'UB2UB_ND2NZ_CPCT'      : PIPE.V,
    # mte3
    'UB2GM'                 : PIPE.MTE3,
    'UB2L1'                 : PIPE.MTE3,
    'UB2L1_NZ'              : PIPE.MTE3,
    'UB2L1_ND2NZ'           : PIPE.MTE3,
    # sort 
    'SORT32'                : PIPE.V,
    'MERGESORT4'            : PIPE.V,
    # # 910B - Vector computations 
    # vector masks 
    'SETMASK'               : PIPE.V,
    'RESETMASK'             : PIPE.V,
    # unary 
    'EXP'                   : PIPE.V,
    'LN'                    : PIPE.V,
    'ABS'                   : PIPE.V,
    'REC'                   : PIPE.V,
    'SQRT'                  : PIPE.V,
    'RSQRT'                 : PIPE.V,
    'VNOT'                  : PIPE.V,
    'RELU'                  : PIPE.V,
    'NOT'                   : PIPE.V,
    # binary 
    'ADD'                   : PIPE.V,
    'SUB'                   : PIPE.V,
    'MUL'                   : PIPE.V,
    'DIV'                   : PIPE.V,
    'MAX'                   : PIPE.V,
    'MIN'                   : PIPE.V,
    'AND'                   : PIPE.V,
    'OR'                    : PIPE.V,
    # unaryscalar
    'ADDS'                  : PIPE.V,
    'MULS'                  : PIPE.V,
    'MAXS'                  : PIPE.V,
    'MINS'                  : PIPE.V,
    'LRELU'                 : PIPE.V,
    'AXPY'                  : PIPE.V,
    # compare 
    'COMPARE'               : PIPE.V,
    'COMPARES'              : PIPE.V,
    'COMPARETOREG'          : PIPE.V,
    'SETCMPMASK'            : PIPE.V,
    'SELECT'                : PIPE.V,
    # cast 
    'CAST'                  : PIPE.V,
    # group
    'VCADD'                 : PIPE.V,
    'VCGADD'                : PIPE.V,
    'VCPADD'                : PIPE.V,
    'VCMAX'                 : PIPE.V,
    'VCGMAX'                : PIPE.V,
    'VCMIN'                 : PIPE.V,
    'VCGMIN'                : PIPE.V,
    # dup brcb 
    'DUP'                   : PIPE.V,
    'BRCB'                  : PIPE.V,
    # gather scatter 
    'GATHER'                : PIPE.V,
    'SCATTER'               : PIPE.V,
    # call micro 
    'CALLMICRO'             : PIPE.V,
}

def check_pipe(i: Instruction):
    if i._inst in PIPE_MAPPING:
        return PIPE_MAPPING[i._inst]
    if 'L1' in i._inst:
        return PIPE.MTE2
    if 'L0C' in i._inst:
        return PIPE.FIX
    if 'L0' in i._inst:
        return PIPE.MTE1
    if 'MAD' in i._inst:
        return PIPE.M
    return None 


def parse_inner(inst_list: list[Instruction], mode: Literal['cube', 'vec']):
    node = InstNode(inst_list, mode, is_root=True)
    # print(node.get_inst())
    return node.get_inst()


class InstNode():
    def __init__(self, inst_list: list[Instruction], mode: Literal['cube', 'vec'], is_root: bool=False):
        if len(inst_list)==0:
            self.parsed = []
            self.head = None 
            return 
        self.check_validity(inst_list, mode)
        if mode=='vec':
            inst_list = self.insert_barv(inst_list)
        self.parsed = []
        self.tmp = []
        self.head = None
        self.used_pipes = set()

        if (inst_list[0]._inst in ['STARTLOOP' , 'STARTIF', 'STARTELIF', 'STARTELSE']) and not is_root:
            self.head = inst_list[0]
            inst_list = inst_list[1:]
        target = self.parsed

        nest_depth = 0
        for i in inst_list:
            if i._inst in ['STARTLOOP' , 'STARTIF', 'STARTELIF', 'STARTELSE']:
                if nest_depth==0:
                    target = self.tmp
                target.append(i)
                nest_depth += 1 
            elif i._inst in ['ENDLOOP', 'ENDIF']:
                nest_depth -= 1 
                if nest_depth==0:
                    target = self.parsed
                    new_node = InstNode(self.tmp, mode)
                    self.tmp = []
                    self.parsed.append(new_node)
                else:
                    target.append(i)
            else:
                target.append(i)
        
        for i in self.parsed:
            self.used_pipes = self.used_pipes.union(pipe(i))

        # insert events 
        if mode=='cube':
            self.process_l1()
            self.process_l0()
            self.process_out()
        elif mode=='vec':
            self.process_vecin()
            self.process_vecout()

    def check_validity(self, inst_list: list[Instruction], mode: Literal['cube', 'vec']):
        def prioritize(i: Instruction):
            if mode=='vec':
                for p in [PIPE.MTE3, PIPE.V, PIPE.MTE2]:
                    if p in pipe(i):
                        return p 
            if mode=='cube':
                for p in [PIPE.FIX, PIPE.M, PIPE.MTE1, PIPE.MTE2]:
                    if p in pipe(i):
                        return p 
            raise Exception()
        
        def is_successor(i: Instruction, prev: PipeInst):
            succ_dict = {
                PIPE.MTE2: [PIPE.V, PIPE.MTE1],
                PIPE.MTE1: [PIPE.M],
                PIPE.M : [PIPE.FIX],
                PIPE.V : [PIPE.MTE3]
            }
            for p in pipe(i):
                if prev not in succ_dict:
                    continue 
                if p in succ_dict[prev]:
                    return True 
            return False 

        curr_pipe = None 
        must_change = False 
        for i in inst_list:
            if len(pipe(i))==1 and None in pipe(i):
                continue 
            if curr_pipe is not None:
                if must_change:
                    if not is_successor(i, curr_pipe):
                        raise Exception('Same pipe must be in the same loop scope')
                else:
                    if not is_successor(i, curr_pipe):
                        if len(pipe(i))>1:
                            raise Exception('Same pipe must be in the same loop scope')
                        else:
                            if list(pipe(i))[0]!=curr_pipe:
                                raise Exception('Same pipe must be in the same loop scope')
            curr_pipe = prioritize(i)
            if len(pipe(i))>1:
                must_change = True 
            else:
                must_change = False 

    def insert_barv(self, inst_list: list[Instruction]):
        res = []
        for i in inst_list:
            res.append(i)
            if len(pipe(i))==1 and PIPE.V in pipe(i):
                res.append(Instruction('BAR', pipe=PIPE.V))
        return res 

    def process_l1(self):
        need_to_insert = True 
        if not (PIPE.MTE2 in self.used_pipes and PIPE.MTE1 in self.used_pipes):
            need_to_insert = False
        for i in self.parsed:
            p = pipe(i)
            if PIPE.MTE2 in p and PIPE.MTE1 in p:
                need_to_insert = False
            
        if need_to_insert:
            for idx in range(len(self.parsed)):
                p = pipe(self.parsed[idx])
                if PIPE.MTE2 in p:
                    break 
            self.parsed.insert(idx, Instruction('EVENTWAIT', name='l1_empty'))

            for idx in range(len(self.parsed)):
                p = pipe(self.parsed[idx])
                if PIPE.MTE1 in p:
                    break 
            self.parsed.insert(idx, Instruction('EVENTWAIT', name='l1_ready'))
            self.parsed.insert(idx, Instruction('EVENTSET', name='l1_ready'))

            has_next = False
            for idx in range(len(self.parsed)):
                p = pipe(self.parsed[idx])
                if PIPE.M in p:
                    has_next = True
                    break 
            if has_next:
                self.parsed.insert(idx, Instruction('EVENTSET', name='l1_empty'))
            else:
                self.parsed.append(Instruction('EVENTSET', name='l1_empty'))

    def process_l0(self):
        need_to_insert = True 
        if not (PIPE.M in self.used_pipes and PIPE.MTE1 in self.used_pipes):
            need_to_insert = False
        for i in self.parsed:
            p = pipe(i)
            if PIPE.M in p and PIPE.MTE1 in p:
                need_to_insert = False
            
        if need_to_insert:
            for idx in range(len(self.parsed)):
                p = pipe(self.parsed[idx])
                if PIPE.MTE1 in p:
                    break 
            self.parsed.insert(idx, Instruction('EVENTWAIT', name='l0_empty'))

            for idx in range(len(self.parsed)):
                p = pipe(self.parsed[idx])
                if PIPE.M in p:
                    break 
            self.parsed.insert(idx, Instruction('EVENTWAIT', name='l0_ready'))
            self.parsed.insert(idx, Instruction('EVENTSET', name='l0_ready'))

            has_next = False
            for idx in range(len(self.parsed)):
                p = pipe(self.parsed[idx])
                if PIPE.FIX in p:
                    has_next = True 
                    break 
            if has_next:
                self.parsed.insert(idx, Instruction('EVENTSET', name='l0_empty'))
            else:
                self.parsed.append(Instruction('EVENTSET', name='l0_empty'))
    
    def process_out(self):
        # try l1 pair 
        need_to_insert = True 
        if not (PIPE.M in self.used_pipes and PIPE.FIX in self.used_pipes):
            need_to_insert = False
        for i in self.parsed:
            p = pipe(i)
            if PIPE.M in p and PIPE.FIX in p:
                need_to_insert = False
            
        if need_to_insert:
            for idx in range(len(self.parsed)):
                p = pipe(self.parsed[idx])
                if PIPE.M in p:
                    break 
            self.parsed.insert(idx, Instruction('EVENTWAIT', name='out_empty'))

            for idx in range(len(self.parsed)):
                p = pipe(self.parsed[idx])
                if PIPE.FIX in p:
                    break 
            self.parsed.insert(idx, Instruction('EVENTWAIT', name='out_ready'))
            self.parsed.insert(idx, Instruction('EVENTSET', name='out_ready'))
            self.parsed.append(Instruction('EVENTSET', name='out_empty'))

    def process_vecin(self):
        # try l1 pair 
        need_to_insert = True 
        if not (PIPE.V in self.used_pipes and PIPE.MTE2 in self.used_pipes):
            need_to_insert = False
        for i in self.parsed:
            p = pipe(i)
            if PIPE.V in p and PIPE.MTE2 in p:
                need_to_insert = False
            
        if need_to_insert:
            for idx in range(len(self.parsed)):
                p = pipe(self.parsed[idx])
                if PIPE.MTE2 in p:
                    break 
            self.parsed.insert(idx, Instruction('EVENTWAIT', name='in_empty'))

            for idx in range(len(self.parsed)):
                p = pipe(self.parsed[idx])
                if PIPE.V in p:
                    break 
            self.parsed.insert(idx, Instruction('EVENTWAIT', name='in_ready'))
            self.parsed.insert(idx, Instruction('EVENTSET', name='in_ready'))

            has_next = False
            for idx in range(len(self.parsed)):
                p = pipe(self.parsed[idx])
                if PIPE.MTE3 in p:
                    has_next = True 
                    break 
            if has_next:
                self.parsed.insert(idx, Instruction('EVENTSET', name='in_empty'))
            else:
                self.parsed.append(Instruction('EVENTSET', name='in_empty'))

    def process_vecout(self):
        # try l1 pair 
        need_to_insert = True 
        if not (PIPE.V in self.used_pipes and PIPE.MTE3 in self.used_pipes):
            need_to_insert = False
        for i in self.parsed:
            p = pipe(i)
            if PIPE.V in p and PIPE.MTE3 in p:
                need_to_insert = False
            
        if need_to_insert:
            for idx in range(len(self.parsed)):
                p = pipe(self.parsed[idx])
                if PIPE.V in p:
                    break 
            self.parsed.insert(idx, Instruction('EVENTWAIT', name='out_empty'))

            for idx in range(len(self.parsed)):
                p = pipe(self.parsed[idx])
                if PIPE.MTE3 in p:
                    break 
            self.parsed.insert(idx, Instruction('EVENTWAIT', name='out_ready'))
            self.parsed.insert(idx, Instruction('EVENTSET', name='out_ready'))
            self.parsed.append(Instruction('EVENTSET', name='out_empty'))

    def get_inst(self):
        res = []
        if self.head is not None:
            res.append(self.head)
        for i in self.parsed:
            if isinstance(i, Instruction):
                res.append(i)
            elif isinstance(i, InstNode):
                res.extend(i.get_inst())
            else:
                raise Exception()
        if self.head is not None:
            if self.head._inst=='STARTLOOP':
                res.append(Instruction('ENDLOOP'))
            else:
                res.append(Instruction('ENDIF'))
        return res 


def pipe(i):
    if isinstance(i, Instruction):
        res = set()
        res.add(check_pipe(i))
        return res 
    elif isinstance(i, InstNode):
        return i.used_pipes
    else:
        raise Exception()

