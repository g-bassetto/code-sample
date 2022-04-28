from abc import ABCMeta, abstractmethod
from .state import PipelineState


class PipelineObject(metaclass=ABCMeta):
    """
    A generic pipeline object, storing a symbolic graph or collection thereof.
    """
    @property
    @abstractmethod
    def compiled(self) -> bool:
        """
        Whether the underlying graph has been compiled.
        """
        raise NotImplementedError()

    @abstractmethod
    def compile(self) -> None:
        """
        Compile the underlying graph.
        """
        pass

    @abstractmethod
    def execute(self) -> None:
        """
        Execute the compiled graph.
        """
        pass


class Function(PipelineObject):
    """
    Wraps a theano Function into a pipeline object.

    All setter methods return a reference to self, allowing to chain them one
    after another.
    """
    inputs: list = list()
    output = None
    givens: dict = None  # a dictionary of variables to substitute
    updates: dict = None  # dictionary of shared updates to perform

    def __setitem__(self, var: str, value) -> None:
        """
        Assign the specified value to the desired slot of the underlying
        Function object.
        :var var: the name of the variable to set
        :var value: float or array_like; the value of the variable
        """
        self._compiled_func[var] = value

    def __getitem__(self, var: str):
        """
        Return the value of a specified variable.
        :param var: the name of the variable to retrieve
        :return: float or array_like; the value of the variable
        """
        return self._compiled_func[var]

    @property
    def compiled(self):
        """
        Whether the underlying graph has been compiled.
        """
        return hasattr(self, '_compiled_func')

    def set_inputs(self, value: list) -> "Function":
        """
        Set the inputs of the underlying graph.
        :var value: a list of symbolic variables
        """
        self.inputs = value
        return self

    def set_output(self, value) -> "Function":
        """
        Set the outputs of the underlying graph.
        :var value: a (list of) symbolic variables this function must compute
        """
        self.output = value
        return self

    def set_givens(self, value) -> "Function":
        """
        Set a list of expressions to substitute in place of some symbolic
        variable.
        :param value: a dict or a list of tuples associating each variable
        with the desired expression
        """
        self.givens = value
        return self

    def set_updates(self, value) -> "Function":
        """
        Set a list of shared variables updates this function must perform.
        :param value: a dict or a list of tuples associating each
        shared variable with the desired expression
        """
        self.updates = value

    def compile(self) -> None:
        """
        Compiles the underlying symbolic graph and cache the results.
        """
        from theano import function
        func = function(
            self.inputs,
            givens=self.givens,
            updates=self.updates,
            on_unused_input='ignore',
            allow_input_downcast=True)
        self._compiled_func = func

    def execute(self):
        """
        Execute the stored theano function.
        """
        self._compiled_func()


class PipelineStep(PipelineObject):
    """
    A PipelineStep is an aggregate of `Function` objects.

    One function is responsible to setup the computational graph, whereas the
    second execute the graph on its input. This is done so that buffers are
    properly initialized for operations relying on them (e.g., the delay
    memory units of a linear filter)

    """

    # stores a reference to the actual binding between the model parameters
    # and the symbolic parameters of the underlying functions
    params: "ParamsBinder"

    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline
        pipeline._steps.append(self)
        self.init = Function()
        self.main = Function()

    @property
    def compiled(self) -> bool:
        """
        Whether the underlying graphs have been compiled.
        """
        return self.init.compiled and self.main.compiled

    def compile(self):
        """
        Compiles the underlying functions if needed.
        """
        if self.compiled:
            return
        if not self.init.compiled:
            self.init.compile()
        if not self.main.compiled:
            self.main.compile()

    def execute(self):
        """
        Execute the stored theano function.
        """
        self.params.setup(self.init)
        self.init.execute()
        self.params.setup(self.main)
        self.main.execute()


class Pipeline:
    """
    A pipeline represents an ordered sequence of operations to apply to a
    signal.
    """
    @property
    def state(self):
        return self._state

    def __init__(self, state):
        self._state = state
        self._steps = list()
        self._maker = PipelineBuilder(self)

    def compile(self):
        """
        Compile the pipeline.
        """
        for step in self._steps:
            step.compile()

    def execute(self):
        """
        Execute the symbolic graph associated to this pipeline.
        """
        for step in self._steps:
            step.execute()

    def __enter__(self):
        return self._maker

    def __exit__(self, exc_type, ex_value, traceback):
        pass


class ParamsBinder:
    """
    Binds a symbolic function's input parameters to actual model parameters.
    """

    def __init__(self, op):
        self.op = op

    def __iter__(self):
        pairs = zip(
            self.op.abstract_params,
            self.op.concrete_params,
        )
        return ((p, v) for p, v in pairs)

    def setup(self, func: Function) -> None:
        """
        Set a function's input parameters to the desired value.
        :param func: the target function
        """
        for p, v in self:
            func[p] = v

    @staticmethod
    def bind(p, v) -> "theano.In":
        """
        Bind a parameter to the value of a shared variable.
        :param p: the target symbolic parameter
        :param v: a shared variable storing the parameter's value
        """
        from theano import In
        return In(p, value=v, implicit=True, allow_downcast=True)


class Operation (metaclass=ABCMeta):
    """
    An abstract operation to perform on a graph.
    """

    # whether we want to compute the output of a recursive filter a single
    # step at a time
    single_step_mode = False

    @property
    @abstractmethod
    def abstract_params(self):
        """
        The input parameters of the specific symbolic sub-graph this
        operation represents.
        """
        pass

    @property
    @abstractmethod
    def concrete_params(self):
        """
        The actual model parameters (or functions thereof) associated to to
        each abstract paramete.
        """
        pass

    @property
    def params(self) -> ParamsBinder:
        """
        Bindings between the operation's parameters and the actual model's
        parameters.
        """
        return ParamsBinder(self)

    @property
    def inputs(self) -> list:
        """
        A list containing the symbolic parameters of this operation.
        """
        return list(self.abstract_params)

    def create_step(self, pipeline) -> PipelineStep:
        """
        Create a new PipelineStep for the desired Pipeline.
        :param pipeline: the target pipeline
        """
        return PipelineStep(pipeline)

    def main(self, signal, template=None):
        """
        The main body of the operation.
        :param signal: the input signal
        :param template:
        :return:
        """
        # build the appropriate step according to the desired
        # single-step mode
        if self.single_step_mode:
            return self.build_step(signal, template)
        else:
            return self.build_loop(signal, template)

    def init(self, signal, template=None):
        return None, []

    @abstractmethod
    def build_step(self, signal, template=None):
        """
        The symbolic graph associated to a single application of a recursive
        filter.
        """
        raise NotImplementedError()

    @abstractmethod
    def build_loop(self, signal, template=None):
        """
        The symbolic graph associated to the application of a recursive filter.
        """
        raise NotImplementedError()


class PipelineBuilder:
    """
    This class is responsible for some manipulations done to a Pipeline
    object, like adding a new step or storing a checkpoint.
    """
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self._state = None
        # checkpoints are a list of dicts mapping this object's state to a
        # particular state of the pipeline
        # they can be used to keep track of shared variable updates
        self.checkpoints = list()

    @property
    def state(self) -> PipelineState:
        """
        The object's state.
        """
        return self._state

    @state.setter
    def state(self, value: PipelineState) -> None:
        """
        Store a specific state. An exception is raised if `value` has the
        wrong number of dymensions.
        :param value: the desired state.
        """
        assert value.ndim == self.pipeline.state.ndim
        self._state = value

    def get_checkpoint(self, index: int):
        """
        Return the desired checkpointed state.
        :param index: index of the desired checkpoint.
        :return: a dict or None if a wrong index is supplied
        """
        try:
            return self.checkpoints[index]
        except IndexError:
            return None

    def set_checkpoint(self, value=None):
        """
        Save a checkpoint to the current state of the pipeline.
        :param value:
        :return:
        """
        if value is not None:
            self.state = value
        self.checkpoints.append({self.state: self.pipeline.state})

    def get_step(self, op: Operation) -> PipelineStep:
        """
        Create a PipelineStep using the Pipeline tracked by this object.
        :param op: the operation we want to add to the pipeline
        :return: the new PipelineStep
        """
        return op.create_step(self.pipeline)

    def add_step(self, op: Operation) -> PipelineStep:
        """
        Add a new step to the pipeline.
        :param op: the operation implemented by the new step
        :return: the new PipelineStep associated with the operation
        """
        from theano.tensor import cast

        newstep = self.get_step(op)
        params = ParamsBinder(op)
        inputs = [params.bind(*pair) for pair in params]

        op.setup(self.state)

        # build the step's initializer
        result, updates = op.init(self.state, self.pipeline.state)
        newstep.init.updates = updates
        newstep.init.inputs = inputs

        # build the step's main body
        result, updates = op.main(self.state, self.pipeline.state)
        updates[self.pipeline.state] = cast(result, self.pipeline.state.dtype)
        newstep.main.inputs = inputs
        # the givens field stores the symbolic result of the application of
        # the previous step - it can be used to run only a certain portion of
        # the pipeline
        newstep.main.givens = self.get_checkpoint(-1)
        newstep.main.updates = updates

        # store a checkpoint to the symbolic variable representing the
        # current state of the pipeline - it can be used by the next step to
        # bypass the previous ones
        self.set_checkpoint(result)

        newstep.params = params

        return newstep
