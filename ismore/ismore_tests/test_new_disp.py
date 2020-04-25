from riglib.stereo_opengl.primitives import Circle, Sector, Line
from riglib.stereo_opengl.primitives import Shape2D


COLORS = {
    'black': (0, 0, 0, 1),
    'red':   (1, 0, 0, 1),
    'green': (0, 1, 0, 1),
    'blue':  (0, 0, 1, 1),
    'white': (1, 1, 1, 1),
    'magenta': (0, 1, 0, 0), 
    'brown': (29, 74, 100, 24),
    'yellow': (0, 0, 1, 0),
}

xy_cursor = Circle(np.array([0., 0.]), 0.5, COLORS['red'])

import pygame

background  = (1,1,1,1)
pygame.init()
window_size = (50, 100)
sub_window_size = (25, 50)
flags = pygame.NOFRAME
screen = pygame.display.set_mode(window_size, flags)
screen_background = pygame.Surface(screen.get_size()).convert()
screen_background.fill(background)


TRANSPARENT = (255,0,255)

surf={}
surf['0'] = pygame.Surface(screen.get_size())
surf['0'].fill(TRANSPARENT)
surf['0'].set_colorkey(TRANSPARENT)

surf['1'] = pygame.Surface(screen.get_size())
surf['1'].fill(TRANSPARENT)
surf['1'].set_colorkey(TRANSPARENT)        

 #values of alpha: higher = less translucent
surf['0'].set_alpha(170) #Cursor
surf['1'].set_alpha(130) #Targets

surf_background = pygame.Surface(surf['0'].get_size()).convert()
surf_background.fill(TRANSPARENT)

pygame.display.update()

color = tuple(map(lambda x: int(255*x), model.color[0:3]))

if isinstance(model, Sphere):
    pos = model._xfm.move[[0,2]]
    pix_pos = self.pos2pix(pos)
    
    rad = model.radius
    pix_radius = self.pos2pix(np.array([model.radius, 0]))[0] - self.pos2pix([0,0])[0]

    #Draws cursor and targets on transparent surfaces
    pygame.draw.circle(self.get_surf(), color, pix_pos, pix_radius)

def draw_world(self):
    #Refreshes the screen with original background
    self.screen.blit(self.screen_background, (0, 0))
    self.surf['0'].blit(self.surf_background,(0,0))
    self.surf['1'].blit(self.surf_background,(0,0))
    
    # surface index
    self.i = 0

    for model in self.world.models:
        self.draw_model(model)
        self.i += 1

    #Renders the new surfaces
    self.screen.blit(self.surf['0'], (0,0))
    self.screen.blit(self.surf['1'], (0,0))
    pygame.display.update()
