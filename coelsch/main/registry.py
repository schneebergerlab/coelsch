from functools import wraps
import click


class OptionRegistry:

    def __init__(self, subcommands):
        self.subcommands = subcommands
        self.register = {}
        self.callback_register = {}

    def register_param(self, click_param, args, kwargs):
        subcommands = kwargs.pop('subcommands', None)
        if subcommands is None:
            raise ValueError('need to provide at least one subcommand to attach option to')

        name = args[-1].split('/')[0].strip('-').replace('-', '_')
        opt = click_param(*args, **kwargs)

        for sc in subcommands:
            if sc not in self.subcommands:
                raise ValueError(f'subcommand {sc} is not pre-registered')
            if sc not in self.register:
                self.register[sc] = {}
            self.register[sc][name] = opt

    def argument(self, *args, **kwargs):
        self.register_param(click.argument, args, kwargs)

    def option(self, *args, **kwargs):
        self.register_param(click.option, args, kwargs)

    def register_callback(self, callback, subcommands=None):
        if subcommands is None:
            subcommands = self.subcommands
        for sc in subcommands:
            if sc not in self.subcommands:
                raise ValueError(f'subcommand {sc} is not pre-registered')
            self.callback_register.setdefault(sc, []).append(callback)

    def callback(self, subcommands=None):
        def _register_callback(callback):
            self.register_callback(callback, subcommands)
        return _register_callback

    def __call__(self, subcommand):
        def _apply_options(func):
            @wraps(func)
            def _wrapped_func(**kwargs):
                for callback in self.callback_register.get(subcommand, [])[::-1]:
                    kwargs = callback(kwargs)
                return func(**kwargs)

            for option in reversed(self.register[subcommand].values()):
                _wrapped_func = option(_wrapped_func)
            return _wrapped_func
        return _apply_options

    def get_kwarg_subset(self, subcommand, kwargs):
        sc_options = list(self.register[subcommand].keys())
        return {kw: val for kw, val in kwargs.items() if kw in sc_options}
