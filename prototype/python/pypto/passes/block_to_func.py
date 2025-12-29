from .. import pypto_impl
from pypto.builder import get_default_builder


class BlockToFunc(pypto_impl.Pass):
    def __init__(self, name: str = "BlockToFunction"):
        super().__init__(name)
        self.liveness = pypto_impl.LivenessAnalyzer()
        self.builder = get_default_builder()
        self.dataflow_num = 0
        
    def RunOnModule(self, module: pypto_impl.ProgramModule) -> pypto_impl.Result:
        for func in module.GetFunctions():
            self.RunOnFunction(func)
        
        return pypto_impl.Result.Success
            
    def RunOnFunction(self, func: pypto_impl.Function) -> pypto_impl.Result:
        # liveness analysis for create dataflow function signature
        self.liveness.RunLivenessAnalysis(func)
        
        # entry function scope and process statement
        scope_guard = self.builder._impl.EnterFunctionBody(func)
        for stmt in self.builder.get_current_scope().GetStatements():
            self.RunOnStatement(stmt)
            
        del scope_guard
            
        return pypto_impl.Result.Success
    
    def RunOnStatement(self, stmt: pypto_impl.Statement) -> pypto_impl.Result:
        if isinstance(stmt, pypto_impl.IfStatement):
            # enter if then scope
            scope_guard = self.builder._impl.EnterIfThen(stmt)
            
            # process then scope stmt
            for then_stmt in self.builder.get_current_scope().GetStatements():
                self.RunOnStatement(then_stmt)
                
            # exit then scope
            del scope_guard
            
            # enter if else scope
            scope_guard = self.builder._impl.EnterIfElse(stmt)
            
            # process else scope stmt
            for else_stmt in self.builder.get_current_scope().GetStatements():
                self.RunOnStatement(else_stmt)
                
            # exit then scope
            del scope_guard
        
        if isinstance(stmt, pypto_impl.ForStatement):
            # enter for scope
            scope_guard = self.builder._impl.EnterForBody(stmt)
            
            # process for scope stmt
            for for_stmt in self.builder.get_current_scope().GetStatements():
                self.RunOnStatement(for_stmt)
            
            # exit then scope
            del scope_guard
            
        if isinstance(stmt, pypto_impl.BlockStatement): 
            # analysis input and output args of block
            # as value is SSA, in/out cast can be simply computed as:
            # in = use - def
            # out = def ∩ liveout
            use_ = self.liveness.GetUse(stmt)
            def_ = self.liveness.GetDef(stmt)
            live_out_ = self.liveness.GetLiveOut(stmt)
            input_args = list(use_.difference(def_))
            output_args = list(def_.intersection(live_out_))
            
            # Cerate new value params for function
            # TODO: Should input and output be added to the scope ?
            input_params = [self.builder._impl.DuplicateValue(arg) 
                            for arg in input_args]
            output_params = [self.builder._impl.DuplicateValue(arg)
                            for arg in output_args]
            
            # Create FunctionSignature
            signature = pypto_impl.FunctionSignature()
            signature.arguments = input_params
            signature.results = output_params
            
            # Create C++ Function using IRBuilder
            cpp_func = self.builder._impl.CreateFunction(
                "dataflow_" + str(self.dataflow_num),
                pypto_impl.FunctionKind.DataFlow,
                signature,
                False
            )
            
            # Enter dataflow function to build statement
            scope_guard = self.builder._impl.EnterFunctionBody(cpp_func)
            
            # TODO: How to make the envTable for this funcion's scope ?
            # IRbuild should give a interface to add exist statement and mainten scope ?
            cpp_func.AddStatement(stmt)
            
            # replace args with params in block
            for arg, param in zip(input_args + output_args, input_params + output_params):
                self.builder._impl.ReplaceValueInBlock(stmt, arg, param)
            
            # create return block
            self.builder._impl.CreateReturn(output_params)
            
            # exit dataflow function, back to control flow
            del scope_guard
            
            # replace block with statement call
            idx = self.builder._impl.RemoveBlockStatement(stmt)
            self.builder._impl.CreateCall("@dataflow_" + str(self.dataflow_num), 
                                          input_args, output_args, idx)
            self.dataflow_num += 1
        else:
            pass
        
        return pypto_impl.Result.Success
        
    def RunOnOperation(self, op_ptr) -> pypto_impl.Result:
        print("op")
        return pypto_impl.Result.Success