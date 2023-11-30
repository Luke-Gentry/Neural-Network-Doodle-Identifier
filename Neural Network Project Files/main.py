import math
import operator
import random
import pygame
from sys import exit
import nn
from keras.datasets import mnist
import numpy as np

pygame.init()
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 800

# GRID INFORMATION:
DIMENSION = 28
PADDING = 10
PIXEL_SIZE = 20
PIXEL_GRID = [[0 for i in range(DIMENSION)] for j in range(DIMENSION)]
COLOR_GRID = [[0 for i in range(DIMENSION)] for j in range(DIMENSION)]

# COLORS
BLACK = (0, 0, 0)
WHITE = (200, 200, 200)




doodle_filenames = ["Draw_Data/Angel.npy", "Draw_Data/Bus.npy", "Draw_Data/Traffic.npy", 
                    "Draw_Data/Tower.npy", "Draw_Data/Skull.npy", "Draw_Data/Spider.npy"]

doodle_names = ["Angel", "Bus", "Traffic Light", "Eiffel Tower", "Skull", "Spider"]

train_X = []
train_y = []
test_X = []
test_y = []

for i in range(len(doodle_filenames)):
    curr_X = np.load(doodle_filenames[i])
    curr_y = [i for j in range(len(curr_X))]
    curr_train_X = curr_X[0:math.floor(len(curr_X) * .9)]
    curr_train_y = curr_y[0:math.floor(len(curr_y) * .9)]
    curr_test_X = curr_X[math.floor(len(curr_X) * .9):len(curr_X) - 1]
    curr_test_y = curr_y[math.floor(len(curr_y) * .9):len(curr_y) - 1]

    for k in range(len(curr_train_X)):
        train_X.append(curr_train_X[k])
        train_y.append(curr_train_y[k])
    
    for k in range(len(curr_test_X)):
        test_X.append(curr_test_X[k])
        test_y.append(curr_test_y[k])


for i in range(len(train_X)):
    train_X[i] = np.where(train_X[i]<100,0,train_X[i])
    train_X[i] = np.where(train_X[i]!=0,255,0)
for i in range(len(test_X)):
    train_X[i] = np.where(test_X[i]<100,0,test_X[i])
    test_X[i] = np.where(test_X[i]!=0,255,0)
#casie is beatiful

def drawPixels():
    for y in range(DIMENSION):
        for x in range(DIMENSION):
            pixel = pygame.Rect((x + 1) * PIXEL_SIZE, (y + 1) * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE)
            PIXEL_GRID[y][x] = pixel
            if COLOR_GRID[y][x] == 0:
                pygame.draw.rect(screen, BLACK, pixel, 0)
            elif COLOR_GRID[y][x] != 0:
                color_val = COLOR_GRID[y][x]
                pygame.draw.rect(screen, (color_val, color_val, color_val), pixel, 0)



def convertPosToPixel(pos):
    return (int(pos[0] / PIXEL_SIZE) - 1, int(pos[1] / PIXEL_SIZE) - 1)



####========================================================= DRAW INITIAL SETUP =========================================================####


global screen, clock
screen=pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
stats_screen=pygame.Surface((SCREEN_WIDTH/2, SCREEN_HEIGHT)).convert_alpha()
clock=pygame.time.Clock()


def drawBorder():
    for i in range(DIMENSION + 2):
        pixel = pygame.Rect(i * PIXEL_SIZE, 0, PIXEL_SIZE, PIXEL_SIZE) 
        pygame.draw.rect(screen, WHITE, pixel, 0) # DRAW TOP BORDER

        pixel = pygame.Rect(i * PIXEL_SIZE, (DIMENSION + 1) * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE)
        pygame.draw.rect(screen, WHITE, pixel, 0) # DRAW BOTTOM BORDER

    for i in range(DIMENSION):

        pixel = pygame.Rect(0, (i + 1) * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE)
        pygame.draw.rect(screen, WHITE, pixel, 0) # DRAW LEFT BORDER

        pixel = pygame.Rect((DIMENSION + 1) * PIXEL_SIZE, (i + 1) * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE)
        pygame.draw.rect(screen, WHITE, pixel, 0) # DRAW RIGHT BORDER




####============================================================== COLOR GRID ==============================================================####



def updateColorGrid(pos):
    pixel_x = pos[0] % 10
    pixel_y = pos[1] % 10
    inpixel_positions = [pixel_y / 10, pixel_x / 10, 1 - (pixel_y / 10), 1 - (pixel_x / 10)] # up right down left
    coord = convertPosToPixel(pos)
    if (coord[0] >= DIMENSION - 1 or coord[1] >= DIMENSION - 1 or coord[0] <= 0 or coord[1] <= 0):
        return
    COLOR_GRID[coord[1]][coord[0]] = 255 
    #COLOR_GRID[coord[1] - 1][coord[0]] = max(COLOR_GRID[coord[1] - 1][coord[0]], 200 * inpixel_positions[0])
    #COLOR_GRID[coord[1]][coord[0] + 1] = max(COLOR_GRID[coord[1]][coord[0] + 1], 200 * inpixel_positions[1])
    #COLOR_GRID[coord[1] + 1][coord[0]] = max(COLOR_GRID[coord[1] + 1][coord[0]], 200 * inpixel_positions[2])
    #COLOR_GRID[coord[1]][coord[0] - 1] = max(COLOR_GRID[coord[1]][coord[0] - 1], 200 * inpixel_positions[3])

    '''
    COLOR_GRID[coord[1] - 1][coord[0] + 1] = max(COLOR_GRID[coord[1] - 1][coord[0] + 1], 200 * inpixel_positions[0] * inpixel_positions[1])
    COLOR_GRID[coord[1] + 1][coord[0] + 1] = max(COLOR_GRID[coord[1] + 1][coord[0] + 1], 200 * inpixel_positions[1] * inpixel_positions[2])
    COLOR_GRID[coord[1] + 1][coord[0] - 1] = max(COLOR_GRID[coord[1] + 1][coord[0] - 1], 200 * inpixel_positions[2] * inpixel_positions[3])
    COLOR_GRID[coord[1] - 1][coord[0] - 1] = max(COLOR_GRID[coord[1] - 1][coord[0] - 1], 200 * inpixel_positions[3] * inpixel_positions[0])
    '''

def updateColorGridMissed(last, curr):
    lastCoord = convertPosToPixel(last)
    currCoord = convertPosToPixel(curr)
    if lastCoord[0] == currCoord[0] and lastCoord[1] == currCoord[1]:
        return
    
    x_dist = (curr[0] - last[0])
    y_dist = (curr[1] - last[1])
    
    tempX = last[0]
    tempY = last[1]
    steps = 12
    for i in range(steps):
        tempX += (x_dist * 1/steps)
        tempY += (y_dist * 1/steps)
        updateColorGrid((tempX, tempY))
        

def updateColorGridErase(pos):
    coord = convertPosToPixel(pos)
    if (coord[0] >= DIMENSION or coord[1] >= DIMENSION):
        return
    COLOR_GRID[coord[0]][coord[1]] = 0
    if (coord[1] + 1 < DIMENSION):
        COLOR_GRID[coord[0]][coord[1] + 1] = 0
    if (coord[0] + 1 < DIMENSION):
        COLOR_GRID[coord[0] + 1][coord[1]] = 0
    if (coord[1] - 1 > 0):
        COLOR_GRID[coord[0]][coord[1] - 1] = 0
    if (coord[0] - 1 > 0):
        COLOR_GRID[coord[0] - 1][coord[1]] = 0


def resetColorGrid():
    for x in range(DIMENSION):
        for y in range(DIMENSION):
            COLOR_GRID[x][y] = 0



####======================================================= VISUALIZE NEURAL NETWORK =======================================================####

neuron_size = 10
circle_positions = []

def DisplayStats(stats):
    stats *= 100
    stats = np.floor(stats)
    '''
    sorted_stats = []
    stats_labels = []
    print(stats)
    for i in range(9):
        max = 0
        for j in range(len(stats)):
            if (stats[j] > stats[max]):
                max = j
        sorted_stats.append(stats[max])
        stats_labels.append(max)
        stats = np.delete(stats, max)
        print(np.delete(stats, max))


    print(stats)
    print(sorted_stats)
    print(stats_labels)
    '''
    
    

    stats_screen.fill((0, 0, 0))
    font = pygame.font.Font('arial.ttf', 32)

    for i in range(len(doodle_names)):
        color = min(255, 255 * (stats[i] / 50))
        color = max(color, 50)
        text = font.render(str(doodle_names[i] + " = " + str(stats[i])), True, (color, color, color))
        textRect = text.get_rect()
        textRect.center = (300, 140 + 40*i)
        stats_screen.blit(text, textRect)

    

####====================================================== INITIALIZE NEURAL NETWORK ======================================================####



drawBorder()
total_data = 800000
'''

inputs = np.abs(np.ceil(.01 * np.random.randn(total_data,4)))
#print(inputs)
labels = np.zeros((total_data, 1))
for i in range(len(inputs)):
    #print(inputs[i])
    if (((inputs[i][0] == 1 and inputs[i][3] == 1) or (inputs[i][1] == 1 and inputs[i][2] == 1))):
        labels[i][0] = 1
    #print(labels[i])

nn.InitializeNeuralNetworkLayers([4, 3, 2])
nn.RunNeuralNetwork([np.array([1, 0, 0, 1])], [0], 1, 1, 1)
'''

nn.InitializeNeuralNetworkLayers([784, 100, 50, 6])
nn.RunNeuralNetwork(train_X, train_y, total_data, 200, .01)
nn.TestNeuralNetwork(test_X, test_y)




####============================================================== GAME LOOP ==============================================================####



### PRIVATE VARIABLES ###
held = False          ###
eraseToggle = False   ###
#########################



run=True
while (run):

    drawPixels()
    g = np.array(COLOR_GRID).flatten() / 255
    stats = nn.RunNeuralNetworkTest([g])
    DisplayStats(stats)
    #print(g)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.MOUSEBUTTONUP:
            held = False
        if pygame.mouse.get_pressed()[0]:
            try:
                pos=pygame.mouse.get_pos()
                updateColorGrid(pos)
                if held:
                   updateColorGridMissed(last,pos)
                held = True
                last = pos
            except AttributeError:
                pass
        keys=pygame.key.get_pressed()
        if keys[pygame.K_r]:
            resetColorGrid()
        if keys[pygame.K_t]:
            r = random.randint(0, len(train_X))
            for i in range(28):
                for j in range(28):
                    COLOR_GRID[i][j] = train_X[r][28*i+j]

    screen.blit(stats_screen, (SCREEN_WIDTH/2, 0))
    pygame.display.update()








