# import io, time, copy
import base64
# import statistics
import pygal
from pygal.util import alter, decorate
# from pygal.style import (
#         DefaultStyle,
#         DarkStyle,
#         NeonStyle,
#         DarkSolarizedStyle,
#         LightSolarizedStyle,
#         LightStyle,
#         CleanStyle,
#         RedBlueStyle,
#         DarkColorizedStyle,
#         LightColorizedStyle,
#         TurquoiseStyle,
#         LightGreenStyle,
#         DarkGreenStyle,
#         DarkGreenBlueStyle,
#         BlueStyle
# )

x_labels_vertical = [
    "IN0", "IN1", "IN2", "IN3", "IN4", "IN5", "IN6", "IN7", "IN8", "IN9", "IN10", "IN11",
    "M00",
    "OUT0", "OUT1", "OUT2", "OUT3", "OUT4", "OUT5", "OUT6", "OUT7", "OUT8", "OUT9", "OUT10", "OUT11",
    "OUT",
    "TIME"
]

x_labels_vertical_nat = [
    "IN0", "IN1", "IN2", "IN3", "IN4", "IN5", "IN6", "IN7", "IN8", "IN9", "IN10", "IN11",
    "M00",
    "OUT",
    "OUT0", "OUT1", "OUT2", "OUT3", "OUT4", "OUT5", "OUT6", "OUT7", "OUT8", "OUT9", "OUT10", "OUT11",
    "TIME"
]

class SquareDots(pygal.Line):
    def __init__(self, *args, **kwargs):
        super(SquareDots, self).__init__(*args, **kwargs)

    def line(self, serie, rescale=False):
        """Draw the line serie"""
        serie_node = self.svg.serie(serie)
        if rescale and self.secondary_series:
            points = self._rescale(serie.points)
        else:
            points = serie.points
        view_values = list(map(self.view, points))
        if serie.show_dots:
            for i, (x, y) in enumerate(view_values):
                if None in (x, y):
                    continue
                if self.logarithmic:
                    if points[i][1] is None or points[i][1] <= 0:
                        continue
                if (serie.show_only_major_dots and self.x_labels
                        and i < len(self.x_labels)
                        and self.x_labels[i] not in self._x_labels_major):
                    continue

                metadata = serie.metadata.get(i)
                classes = []
                if x > self.view.width / 2:
                    classes.append('left')
                if y > self.view.height / 2:
                    classes.append('top')
                classes = ' '.join(classes)

                self._confidence_interval(
                    serie_node['overlay'], x, y, serie.values[i], metadata
                )

                dots = decorate(
                    self.svg,
                    self.svg.node(serie_node['overlay'], class_="dots"),
                    metadata
                )

                val = self._format(serie, i)
                alter(
                    self.svg.transposable_node(
                        dots,
                        'rect',
                        x=x - serie.dots_size / 2,
                        y=y - serie.dots_size / 2,
                        width=serie.dots_size,
                        height=serie.dots_size,
                        class_='dot reactive tooltip-trigger'
                    ), metadata
                )
                self._tooltip_data(
                    dots, val, x, y, xlabel=self._get_x_label(i)
                )
                self._static_value(
                    serie_node, val, x + self.style.value_font_size,
                    y + self.style.value_font_size, metadata
                )

        if serie.stroke:
            if self.interpolate:
                points = serie.interpolated
                if rescale and self.secondary_series:
                    points = self._rescale(points)
                view_values = list(map(self.view, points))
            if serie.fill:
                view_values = self._fill(view_values)

            if serie.allow_interruptions:
                # view_values are in form [(x1, y1), (x2, y2)]. We
                # need to split that into multiple sequences if a
                # None is present here

                sequences = []
                cur_sequence = []
                for x, y in view_values:
                    if y is None and len(cur_sequence) > 0:
                        # emit current subsequence
                        sequences.append(cur_sequence)
                        cur_sequence = []
                    elif y is None:  # just discard
                        continue
                    else:
                        cur_sequence.append((x, y))  # append the element

                if len(cur_sequence) > 0:  # emit last possible sequence
                    sequences.append(cur_sequence)
            else:
                # plain vanilla rendering
                sequences = [view_values]
            if self.logarithmic:
                for seq in sequences:
                    for ele in seq[::-1]:
                        y = points[seq.index(ele)][1]
                        if y is None or y <= 0:
                            del seq[seq.index(ele)]
            for seq in sequences:
                self.svg.line(
                    serie_node['plot'],
                    seq,
                    close=self._self_close,
                    class_='line reactive' +
                    (' nofill' if not serie.fill else '')
                )

# huge thanks to rouilj from https://github.com/Kozea/pygal/issues/516 for this combined LineBar system
class LineBar(pygal.StackedBar, SquareDots):
    def __init__(self, config=None, **kwargs):
        super(LineBar, self).__init__(config=config, **kwargs)
        self.y_title_secondary = kwargs.get('y_title_secondary')
        self.plotas = kwargs.get('plotas', 'line')

    def _make_y_title(self):
        super(LineBar, self)._make_y_title()

        # Add secondary title
        if self.y_title_secondary:
            yc = self.margin_box.top + self.view.height / 2
            xc = self.width - 10
            text2 = self.svg.node(
                self.nodes['title'], 'text', class_='title',
                x=xc,
                y=yc
            )
            text2.attrib['transform'] = "rotate(%d %f %f)" % (
                -90, xc, yc)
            text2.text = self.y_title_secondary

    def _plot(self):
        for i, serie in enumerate(self.series, 1):
            plottype = self.plotas

            raw_series_params = self.svg.graph.raw_series[serie.index][1]
            if 'plotas' in raw_series_params:
                plottype = raw_series_params['plotas']

            if plottype == 'bar':
                self.bar(serie)
            elif plottype == 'line':
                self.line(serie)
            else:
                raise ValueError('Unknown plottype for %s: %s'%(serie.title, plottype))

        for i, serie in enumerate(self.secondary_series, 1):
            plottype = self.plotas

            raw_series_params = self.svg.graph.raw_series[serie.index][1]
            if 'plotas' in raw_series_params:
                plottype = raw_series_params['plotas']

            if plottype == 'bar':
                self.bar(serie, True)
            elif plottype == 'line':
                self.line(serie, True)
            else:
                raise ValueError('Unknown plottype for %s: %s'%(serie.title, plottype))

class DataPlot():
    def __init__(self):
        self.lower = None
        self.upper = None
        self.data = []
        self.experimental_range = False

    def normalize_score(self, score):
        #nifty trick to allow division by 0 from https://stackoverflow.com/a/68118106
        return (self.upper - self.lower) and (score - self.lower)/(self.upper - self.lower) or 0

    def return_color(self, score, style):
        clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
        if style.colors[2].startswith("#"):
            style_rgb = (*[int(style.colors[2].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)], )
        elif style.colors[2].startswith("rgb"):
            style_rgb = (*[int(i) for i in style.colors[2].lstrip("rgb(").rstrip(")").split(",")], )
        else:
            style_rgb = (0, 0, 0)
        rgb = [round(clamp(n-min(style_rgb)+score*255, 0, 255)) for n in style_rgb]
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 255)"

    def add_data(self, weights, score, style, show_labels):
        # check for experimental range outside of (0, 1)
        if not all(weight >= 0 and weight <= 1 for weight in weights):
            self.experimental_range = True

        # convert to standard from nat
        # weights = weights.copy()
        # out = weights.pop(13)
        # time_embed = weights.pop(-1)
        # weights.append(out)
        # weights.append(time_embed)

        if self.lower is None or self.upper is None:
            self.lower, self.upper = score, score

        if score < self.lower or score > self.upper:
            if score < self.lower:
                self.lower = score
            if score > self.upper:
                self.upper = score
            for serie, stored_score in self.data:
                for point in serie:
                    point["color"] = self.return_color(self.normalize_score(stored_score), style)

        #shift experimental range range up by 1 since pygal doesn't like negative values'
        chart_add = 1 if self.experimental_range else 0
        self.data.append(([{"value": weight + chart_add, "color": self.return_color(self.normalize_score(score), style)} for weight in weights], score))

        css = ['file://style.css', 'file://graph.css', 'inline:#activate-serie-0 {transform: translate(0, 21px);}', 'inline:#activate-serie-1 {transform: translate(0, -21px);}']
        for num in range(len(self.data)):
            css.append(f'inline:g.plot.overlay g.serie-{2+num}, g.graph g.serie-{2+num} ' + '{transform: translate(31px,0);}')
            css.append(f'inline:#activate-serie-{2+num} ' + '{display: none;}')

        chart_range = (0, 3) if self.experimental_range else (0, 1)
        chart = LineBar(css=css, stroke=False, style=style, height=300, width=1820, range=chart_range, secondary_range=chart_range, show_y_labels=show_labels)
        # chart.x_labels = x_labels_vertical
        chart.x_labels = x_labels_vertical_nat

        chart.add("B", [{"value": weight + chart_add, 'ci': {'low': chart_range[0], 'high': chart_range[1]}} for weight in weights], allow_interruptions=False, show_dots=True, dots_size=5, rounded_bars=10, plotas="bar")
        chart.add("A", [1 - weight + chart_add for weight in weights], allow_interruptions=True, show_dots=True, dots_size=5, rounded_bars=10, plotas="bar")

        data_length = len(self.data)
        for idx, (serie, stored_score) in enumerate(self.data):
            #stroke: , stroke=True, stroke_style={'dasharray': '3, 6'}
            #dots_size=round((3-self.normalize_score(stored_score))*3)
            chart.add(str(idx+1), serie, allow_interruptions=True, dots_size=9, plotas="line")

        # before = time.time()
        data = chart.render(pretty_print=True)
        uri = 'data:image/svg+xml;charset=utf-8;base64,' + base64.b64encode(data).decode("utf-8")
        # after = time.time()
        # logger.debug(after-before)

        return f'<img src="{uri}"/>', data
