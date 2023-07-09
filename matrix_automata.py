from typing import Protocol
import numpy as np
from glumpy import app, gloo, gl, key
from rules import CallableRuleset, basic_convolution, fast_inv_gaussian_activation, checkerboard_intervetion

app.use('qt5')


class AutomataRuleset(Protocol):
    def convolution(self, field: np.ndarray) -> np.ndarray:
        ...

    def activation(self, field: np.ndarray) -> np.ndarray:
        ...

    def __call__(self, field: np.ndarray) -> np.ndarray:
        ...

    def intervention(self, field: np.ndarray, x: int, y: int) -> None:
        ...


class Buffer(Protocol):
    def get_smoothed(self) -> np.ndarray:
        ...

    def append(self, field: np.ndarray) -> None:
        ...

class MeanFrameBuffer:
    def __init__(self, size):
        self.size = size
        self._buffer = []

    def get_smoothed(self):
        return np.mean(np.stack(self._buffer), axis=0)

    def append(self, field):
        self._buffer.append(field)
        if len(self._buffer) > self.size:
            self._buffer.pop(0)


class EveryNthBuffer:
    def __init__(self, size):
        self.size = size
        self.state = 0

    def get_smoothed(self):
        return self._buffer

    def append(self, field):
        if self.state == 0:
            self._buffer = field
        self.state += 1
        if self.state >= self.size:
            self.state = 0


class OneFrameFakeBuffer:
    'it just feels morally wrong to do real computations for a 1-frame buffer'

    def __init__(self):
        self._buffer = None

    def get_smoothed(self):
        return self._buffer

    def append(self, field):
        self._buffer = field


class AutomataDisplay:
    def __init__(
        self,
        rules:AutomataRuleset,
        field_size:'tuple[int,int]|float',
        display_size:'tuple[int, int]',
        color:'tuple[np.int8, np.int8, np.int8]',
        buffer:Buffer=None,
        fullscreen=True,
    ):
        self.window = app.Window(*display_size, fullscreen=fullscreen)
        disx, disy = self.window.get_size()
        if isinstance(field_size, (int, float)):
            field_size = (int(disy * field_size), int(disx * field_size))
        self.field = np.random.uniform(0, 1, size=field_size).astype('float32')
        self.rules = rules

        if buffer is None:
            self.buffer = OneFrameFakeBuffer()
        else:
            self.buffer = buffer

        self.quad = gloo.Program(
            vertex='''
                attribute vec2 position;
                attribute vec2 texcoord;
                varying vec2 v_texcoord;
                void main()
                {
                    gl_Position = vec4(position, 0.0, 1.0);
                    v_texcoord = texcoord;
                }
            ''',
            fragment='''
                uniform vec3 color;
                uniform sampler2D texture;
                varying vec2 v_texcoord;
                void main()
                {
                    gl_FragColor = vec4(color * texture2D(texture, v_texcoord).r, 1);
                }
            ''',
            count=4,
        )
        self.quad['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.quad['texcoord'] = [
            (0,  1), (0,  0), (1,  1), (1,  0)]  # fmt: skip
        self.quad['texture'] = self.field
        self.quad['color'] = np.array(color) / 255

        self.pause = False
        self.total_frames = 0
        self.window.event(self.on_draw)
        self.window.event(self.on_key_press)
        self.window.event(self.on_mouse_drag)
        self.window.event(self.on_mouse_press)

    def on_draw(self, dt):
        if not self.pause:
            self.total_frames += 1
            self.field = self.rules(self.field)
            self.buffer.append(self.field)
            self.quad['texture'] = self.buffer.get_smoothed()
            self.window.clear()
            self.quad.draw(gl.GL_TRIANGLE_STRIP)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            print(f'[i] Quitting. FPS was {self.window.fps:.2f}. '
                  f'Simultaion ran for {self.total_frames} frames.'
            )
            print(f'{np.mean(self.field) = }')
            app.quit()

        elif symbol == key.SPACE:
            self.pause = not self.pause

    def mouse_input(self, posx, posy):
        disx, disy = self.window.get_size()
        fieldy, fieldx = self.field.shape
        x = int(posx / disx * fieldx)
        y = int(posy / disy * fieldy)
        try:
            self.rules.intervention(self.field, y, x)
        except (ValueError, IndexError):
            pass

    def on_mouse_drag(self, x, y, dx, dy, button):
        self.mouse_input(x, y)

    def on_mouse_press(self, x, y, button):
        self.mouse_input(x, y)

    def __call__(self, framerate=0):
        app.run(framerate=framerate)

if __name__ == '__main__':

    AutomataDisplay(
        rules=CallableRuleset(
            np.array([[ 0.24879229, -0.8920062 ,  0.24879229],
        [-0.8920062 ,  0.46585773, -0.8920062 ],
        [ 0.24879229, -0.8920062 ,  0.24879229]]),
            basic_convolution,
            fast_inv_gaussian_activation,
            checkerboard_intervetion
        ),
        display_size=(1, 1),
        field_size=1,
        color=(100, 180, 100),
        fullscreen=True,
        buffer=EveryNthBuffer(4),
    )()
