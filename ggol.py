from typing import Protocol
import numpy as np
import pygame
import good_rules


class GOLRuleset(Protocol):
    def __call__(self, field: np.ndarray) -> np.ndarray:
        ...

    def intervention(self, field:np.ndarray, x: int, y: int) -> None:
        ...


class Buffer:

    def __init__(self, color, size):
        self.size = size
        self._buffer = []
        self.color = color

    def append(self, field):
        self._buffer.append(field)
        if len(self._buffer) > self.size:
            self._buffer.pop(0)
    
    def get_smoothed(self):
        if len(self._buffer) == 1:
            field = self._buffer[0]
        else:
            field = np.mean(np.stack(self._buffer), axis=0)
        return (field[..., np.newaxis] * self.color).astype('uint8')
        

class GGOL:

    def __init__(
        self, 
        rules: GOLRuleset, 
        field_size: tuple[int,int], 
        display_size: 'tuple[int, int]|float', 
        color, 
        buffer_size=1,
        display_flags=0
    ):
        pygame.init()
        self.display = pygame.display.set_mode(display_size, display_flags)
        disx, disy = self.display.get_size()
        if isinstance(field_size, (int, float)):
            field_size = (int(disx*field_size), int(disy*field_size))
        self.field = np.random.binomial(1,p=rules.initialization_percentage, size=field_size).astype('float32')
        self.rules = rules
        self.font = pygame.font.SysFont("Arial", 18)
        self.clock = pygame.time.Clock()
        self.buffer = Buffer(np.full(field_size+(3,),color).astype('uint8'), buffer_size)


    def draw(self):
        self.field = self.rules(self.field)

        self.buffer.append(self.field)

        surf = pygame.transform.scale(
                    pygame.surfarray.make_surface(
                        self.buffer.get_smoothed()),
                    self.display.get_size()
        )

        self.display.blit(surf, (0, 0))
        self.clock.tick()
        fps = str(self.clock.get_fps())
        self.display.blit(self.font.render(fps, 1, pygame.Color("coral")), (0, 0))
     

    def __call__(self):

        alive = True
        paused = False
        while alive:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    alive = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = ~paused
                    if (
                        event.key == pygame.K_ESCAPE 
                        or event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_CTRL
                        ):
                        alive = False
                elif event.type in [pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN]:
                    if pygame.mouse.get_pressed()[0]:
                        posx, posy = pygame.mouse.get_pos()
                        disx, disy = self.display.get_size()
                        fieldx, fieldy = self.field.shape
                        x = int(posx/disx*fieldx)
                        y = int(posy/disy*fieldy)
                        try:
                            self.rules.intervention(self.field, x, y)
                        except (ValueError, IndexError):
                            pass

            if not paused:
                self.draw()

            pygame.display.update()

        pygame.quit()



# GGOL(
#     rules=good_rules.slime_pulling_worms, 
#     display_size=(0, 0),
#     field_size=1/2,
#     color=(180,180,100),
#     display_flags=pygame.FULLSCREEN
# )()
