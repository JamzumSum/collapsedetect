from typing import Callable, List, Optional, Type

from torch.nn import Module
from typing_extensions import final

from . import T, is_collapse

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except:
    SummaryWriter = Type


class CollapseDetector:
    handle = None

    def __init__(self) -> None:
        self.store: List[T] = []
        self.clear = self.store.clear

    def append(self, data: T):
        self.store.append(data)
        return self

    @final
    def detect(self, eps=1e-5, clear: bool = True) -> bool:
        assert self.store, "nothing is inside detector!"
        r = is_collapse(*self.store, eps=eps)
        if clear:
            self.clear()
        return r

    @final
    def hook(
        self,
        mod: Module,
        pre: bool = False,
        filter_fn: Optional[Callable[..., T]] = None,
    ):
        assert self.handle is None, "hook can be called only once"

        filter_fn = filter_fn or (lambda x: x)
        if pre:
            hook = lambda _, input: self.append(filter_fn(input)) and None
            self.handle = mod.register_forward_pre_hook(hook)
        else:
            hook = lambda _, _i, output: self.append(filter_fn(output)) and None
            self.handle = mod.register_forward_hook(hook)

    def rm_hook(self):
        assert self.handle, "no hook is registered"
        self.handle.remove()
        self.handle = None



class CollapseVisualizer(CollapseDetector):
    def __init__(self, writer: SummaryWriter, tag: Optional[str] = None) -> None:
        assert self.check_tensorboard(), "tensorboard module not available!"
        super().__init__()
        self.writer = writer
        self.tag = tag or self.__class__.__name__

    def append(self, data: T, **kw):
        self.writer.add_histogram(self.tag, data, **kw)
        return super().append(data)

    @staticmethod
    def check_tensorboard():
        try:
            import tensorboard

            return True
        except ImportError:
            return False
