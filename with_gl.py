import numpy as np
import good_rules
from glumpy import app, gloo, gl, data

app.use('qt5')

rules = good_rules.blood_pumping_worms
window = app.Window(fullscreen=True)
h, w = window.get_size()
field_size = 1/2
if isinstance(field_size, float):
    field_size = (int(w*field_size), int(h*field_size))
field = np.random.binomial(1,p=rules.initialization_percentage, size=field_size).astype('float32')
quad = gloo.Program(
    vertex="""
        attribute vec2 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        void main()
        {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = texcoord;
        }
    """, 
    fragment="""
        uniform vec3 color;
        uniform sampler2D texture;
        varying vec2 v_texcoord;
        void main()
        {
            gl_FragColor = vec4(color * texture2D(texture, v_texcoord).r, 1);
        }
    """,
    count=4)
quad['position'] = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
quad['texcoord'] = [( 0, 1), ( 0, 0), ( 1, 1), ( 1, 0)]
quad['texture'] = field
quad['color'] = np.array([200,10,150]) / 255


@window.event
def on_draw(dt):
    global field
    field = rules(field)
    quad['texture'] = field
    # window.clear()
    quad.draw(gl.GL_TRIANGLE_STRIP)
    print(window.fps)

app.run(framerate=500)
