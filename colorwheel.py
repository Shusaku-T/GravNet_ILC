import numpy as np


class ColorWheel:
    '''Returns a consistent color when given the same object'''
    def __init__(self, colors=None, seed=44, assignments=None, shuffle=True, n=None):
        if colors is None:
            import matplotlib._color_data as mcd
            self.colors = list(mcd.XKCD_COLORS.values())
        elif colors == 'viridis':
            from matplotlib import cm
            from colour import Color
            if n is None: n = 30
            viridis = cm.get_cmap('viridis', n)
            self.colors = [ Color(rgb=viridis(i/float(n))[:-1]).hex for i in range(n) ]
        else:
            self.colors = colors
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.colors)
        self._original_colors = self.colors.copy()
        self.assigned_colors = {}
        if assignments:
            [self.assign(k, v) for k, v in assignments.items()]
        
    def make_key(self, thing):
        try:
            return int(thing)
        except ValueError:
            return thing

    def __contains__(self, thing):
        return self.make_key(thing) in self.assigned_colors

    def __call__(self, thing):
        key = self.make_key(thing)
        if key in self.assigned_colors:
            # print(f'Returning pre-assigned color: {key}:{self.assigned_colors[key]}')
            return self.assigned_colors[key]
        else:
            color = self.colors.pop()
            self.assigned_colors[key] = color
            if not(self.colors): self.colors = self._original_colors.copy()
            # print(f'Returning newly assigned color: {key}:{self.assigned_colors[key]}')
            return color
    
    def assign(self, thing, color):
        """Assigns a specific color to a thing"""
        key = self.make_key(thing)
        self.assigned_colors[key] = color
        if color in self.colors: self.colors.remove(color)

    def many(self, things, color=None):
        for i, t in enumerate(things):
            if color is None and i == 0:
                color = self(t)
            else:
                self.assign(t, color)



class ColorwheelWithProps:
    def __init__(self, *args, **kwargs):
        self.cw = ColorWheel(*args, **kwargs)
        # self.cw.colors = [ Property(c) for c in self.cw.colors ]
        # self.cw._original_colors = self.cw.colors.copy()
        self.assigned_props = {}

    def __call__(self, thing):
        if thing not in self:
            raise ValueError(f'__call__ for ColorwheelWithProps only works for assigned properties; no such key {thing}')
        return self.assigned_props[self.cw.make_key(thing)]

    def __contains__(self, thing):
        return self.cw.make_key(thing) in self.assigned_props

    def assign(self, thing, **kwargs):
        # Make sure there is a color
        if 'color' in kwargs:
            self.cw.assign(thing, kwargs['color'])
        else:
            kwargs['color'] = self.cw(thing)
        # Compile a property dict and save it
        key = self.cw.make_key(thing)
        self.assigned_props[key] = kwargs

    def many(self, things, **props):
        n_things = len(things)
        # Assign a color in props if not given explicitely
        if not 'color' in props: props['color'] = self.cw(things[0])
        # Turn all len-1 props into lists
        for key, val in props.items():
            if isinstance(val, str) or not hasattr(val, '__len__'):
                props[key] = [ val for _ in range(n_things) ]
            assert len(props[key]) == n_things
        # Now assign 1 dict per thing
        for i, thing in enumerate(things):
            self.assign(thing, **{ key : props[key][i] for key in props })




class HighlightColorwheel:
    def __init__(self):
        from colour import Color
        import matplotlib._color_data as mcd
        # Range from light to dark blue
        normal_colors = [c.hex for c in Color('#d4e4ff').range_to(Color('#000263'), 100)]
        # Highlight colors: Anything but blue
        blue_blacklist = [
            'cerulean', 'navy', 'azure', 'cobalt', 'cornflower', 'sky', 'denim',
            'ultramarine', 'marine', 'light navy', 'royal'
            ]
        highlight_colors = [
            c for k, c in mcd.XKCD_COLORS.items() if not 'blu' in k.lower() and k.lower() not in blue_blacklist
            ] + ['purple', 'orange', 'green', 'red']
        self.cw_normal = ColorWheel(colors=normal_colors)
        self.cw_highlight = ColorWheel(colors=highlight_colors, shuffle=False)

    def highlight(self, thing):
        return self.cw_highlight(thing)

    def highlight_many(self, things):
        return self.cw_highlight.many(things)

    def normal(self, thing):
        return self.cw_normal(thing)

    def normal_many(self, things):
        return self.cw_normal.many(things)

    def assign(self, thing, color):
        self.cw_normal.assign(thing, color)
        self.cw_highlight.assign(thing, color)

    def __call__(self, thing):
        if thing in self.cw_highlight:
            return self.cw_highlight(thing)
        else:
            return self.normal(thing)
    


