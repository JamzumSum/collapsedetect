from typing import Callable, List, Optional, Type

from torch.nn import Module
from typing_extensions import final

from . import T, is_collapse

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except:
    SummaryWriter = Type


class CollapseDetector:
    """Basic collapse detector."""

    handle = None

    def __init__(self) -> None:
        self.store: List[T] = []
        """data store"""
        self.clear = self.store.clear
        """Clear :obj:`.store`.
        
        :return: None
        """

    def append(self, data: T):
        """Save the data into the store of the detector. Use :obj:`.clear` to clear all saved data.
        Use :obj:`.detect` to call :meth:`is_collapse` on saved data.

        .. note:: `data` will be detached and transfered to cpu at once.

        :param data: data to be saved
        :return: self
        """
        self.store.append(data.detach().cpu())
        return self

    @final
    def detect(self, eps=1e-5, clear: bool = True) -> bool:
        """Call :meth:`~collapsedetect.is_collapse` on the saved data. This will clear the :obj:`.store` by default.

        :param eps: as that in :meth:`~collapsedetect.is_collapse`.
        :param clear: whether to clear the :obj:`.store` after detection, defaults to True.
        :return: whether saved data is collapse.

        .. seealso:: :meth:`~collapsedetect.is_collapse`
        """
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
        """Add hook for a :external+torch:class:`~torch.nn.Module`, so that its input/output will be appended into
        :obj:`.store` automatically. Hooks can be removed by :meth:`.rm_hook`.

        :param mod: module to be registered
        :param pre: True to register ``forward_pre_hook``, else ``forward_hook``
        :param filter_fn: a getter to the target tensor, given the input(pre=True) or output(pre=False) of a module.

        .. warning::

            Hooks are binded to one detector instance. Means one detector can only register one module. You may
            instantiate another detector or remove current hook if you want to hook another module.

        .. seealso::

            :external+torch:meth:`torch.nn.Module.register_forward_pre_hook`
            :external+torch:meth:`torch.nn.Module.register_forward_hook`

        Example1:

        >>> resnet = get_resnet18()     # This is a module with single output
        >>> det = CollapseDetector()
        >>> det.hook(resnet)
        >>> for _ in range(15): resnet(image)
        >>> det.detect()
        False

        Example2: pre=True

        >>> trainer = get_trainer()     # This is a trainer with network and loss module
        >>> trainer.criterion           # This is a module with single input
        mybug.TripletLoss
        >>> det = CollapseDetector()
        >>> det.hook(trainer.criterion, pre=True)   # Hook the input value, without caring where the input comes from
        >>> det.rm_hook()

        Example3: usage of `filter_fn`

        >>> net = get_pair_net()     # This is a module that return two value: <vector, label>
        >>> net(image)          # will return a tuple with two value
        >>> (torch.Tensor(...), 1)
        >>> det = CollapseDetector()
        >>> det.hook(net, filter_fn=lambda t: t[0])     # filter_fn is a getter that unpacks the pair and returns the first part
        >>> det.rm_hook()
        """
        assert self.handle is None, "hook can be called only once"

        filter_fn = filter_fn or (lambda x: x)
        if pre:
            hook = lambda _, input: self.append(filter_fn(input)) and None
            self.handle = mod.register_forward_pre_hook(hook)
        else:
            hook = lambda _, _i, output: self.append(filter_fn(output)) and None
            self.handle = mod.register_forward_hook(hook)

    def rm_hook(self):
        """Remove hook registered by :meth:`.hook`. You must remove the existing hook
        before you register a new hook using the same detector.
        """
        assert self.handle, "no hook is registered"
        self.handle.remove()
        self.handle = None


class CollapseVisualizer(CollapseDetector):
    def __init__(self, writer: SummaryWriter, tag: Optional[str] = None) -> None:
        """
        :param writer: a tensorboard SummaryWriter to write into
        :param tag: default tag for appended data in tensorboard histogram view, default as current classname.

        .. warning::

            Instantiate this class will trigger tensorboard installation checking.
        """
        assert self.check_tensorboard(), "tensorboard module not available!"
        super().__init__()
        self.writer = writer
        self.tag = tag or self.__class__.__name__
        """default tag for appended data in tensorboard histogram view"""

    def append(self, data: T, **kw):
        """:meth:`CollapseVisualizer.append` will append the data into `.store` as that in :meth:`CollapseDetector.append`.
        It will also add the data into histogram of tensorboard, tagged with :obj:`.tag`.

        .. seealso:: :external+torch:meth:`~torch.utils.tensorboard.writer.SummaryWriter.add_histogram`
        """
        tag = kw.get("tag") or self.tag
        data = data.detach().cpu()
        self.writer.add_histogram(tag, data, **kw)
        return super().append(data)

    @staticmethod
    def check_tensorboard():
        try:
            import tensorboard

            return True
        except ImportError:
            return False
