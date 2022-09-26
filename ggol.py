from typing import Protocol
import cv2
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
        self.color = np.array(color, dtype='uint8')

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
        display_size: tuple[int, int], 
        color, 
        buffer_size=1
    ):
        pygame.init()
        self.field = np.random.binomial(1,p=rules.initialization_percentage, size=field_size).astype('float32')
        self.rules = rules
        self.display = pygame.display.set_mode(display_size)
        self.font = pygame.font.SysFont("Arial", 18)
        self.clock = pygame.time.Clock()
        self.buffer = Buffer(color, buffer_size)
        
    def set_title(self, title):
        pygame.display.set_caption(title)
    
    def start(self):
        textcolor = pygame.Color("coral")
        alive = True
        paused = False
        while alive:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    alive = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = ~paused
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
                self.display.blit(self.font.render(fps, 1, textcolor), (0, 0))

            pygame.display.update()

        pygame.quit()



rules = good_rules.slime_pulling_worms

viewer = GGOL(
    rules=rules, 
    display_size=(800, 800),
    field_size=(800, 800),
    color=(200,200,250)
)
viewer.start()