#Connect 4 by Jamie McMullin (jamai)
#Uses Q-Learning to kind learn Connect 4
import pygame
import time
import random
import numpy as np
import copy
import matplotlib.pyplot as plt

import os, sys, time, datetime, json, random
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers import PReLU, Flatten, Dense, Dropout


#Variables
gameSpeed = 30
gamesPlayed = 1
cellX = 50
cellY = 50
theGridX = 7
theGridY = 6
offset = (cellX + cellY) / 4
circleRad = 20
p1Col = (255,0,0)
p2Col = (255,255,0)
noCol = (255,255,255)
theCol = 0
moves = 0
AIWin = 0
ranWin = 0
gameWon = 0
drawGames = 0
setsPlayed = 0
bestWinPercent = .25
cursPos = 0
winImprove = 0.005

humanPlayer = False
isOver = False
modelSaved = False
builtModel = False
notMoved = False
totalGames = 250
modelVersion = 1
theGrid = list()
memory = list()
roundPlot = list()
scoreRecord = list()

#AI Parameters
dataSize = 10
maxMemory = 50
startEpsilon = 0.9
minEpsilon = 0.1
epsilon = startEpsilon
discount = 0.05
epsilonDecay = 0.005

#PyGame Declarations
pygame.init()
clock = pygame.time.Clock()

dis=pygame.display.set_mode((600, 480)) #Display size

#Set Text Box
fontSize = 20
font_style = pygame.font.SysFont(None, fontSize)
def message(msg, x, y, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [x, y])


def resettheGrid(theGrid):
    theGrid = np.zeros((7,6))
    return theGrid


def buildModel(obs):
    
    model = Sequential () 
    model.add(Dense(128, activation="relu",
                        input_dim=obs.size))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(7, activation="linear"))
    model.compile(loss="mse",
                     optimizer=Adam(lr=.001))
    return model


def getValid(move):
    
    for y in range(theGridY):
        if theGrid[move][y] != 0:
            if y > 0:
                return y-1
            else:
                return -1
    return y


def detectWin(grid, x, y, fit):
    
    gameWon = 0
    
    #Find out which player it is
    thePlayer = 0
    if grid[x][y] == 0.5:
        thePlayer = 0.5
    elif grid[x][y] == 1.0:
        thePlayer = 1
    if fit == True and grid[x][y] == 0.5:
        thePlayer = 1
    elif fit == True and grid[x][y] == 1:
        thePlayer = 0.5
    
    #Test VERTICAL
    if y < 3:
        if grid[x][y+1] == thePlayer and grid[x][y+2] == thePlayer and grid[x][y+3] == thePlayer:
            gameWon = thePlayer
    #Test HORIZONTAL
    if x == 0 and gameWon == 0:
        if grid[x+1][y] == thePlayer and grid[x+2][y] == thePlayer and grid[x+3][y] == thePlayer:
            gameWon = thePlayer
    elif x == 1 and gameWon == 0:
        if grid[x+1][y] == thePlayer and grid[x+2][y] == thePlayer and grid[x+3][y] == thePlayer:
            gameWon = thePlayer
        elif grid[x-1][y] == thePlayer and grid[x+1][y] == thePlayer and grid[x+2][y] == thePlayer:
            gameWon = thePlayer
    elif x == 2 and gameWon == 0:
        if grid[x+1][y] == thePlayer and grid[x+2][y] == thePlayer and grid[x+3][y] == thePlayer:
            gameWon = thePlayer
        elif grid[x-1][y] == thePlayer and grid[x+1][y] == thePlayer and grid[x+2][y] == thePlayer:
            gameWon = thePlayer
        elif grid[x-2][y] == thePlayer and grid[x-1][y] == thePlayer and grid[x+1][y] == thePlayer:
            gameWon = thePlayer
    elif x == 3 and gameWon == 0:
        if grid[x+1][y] == thePlayer and grid[x+2][y] == thePlayer and grid[x+3][y] == thePlayer:
            gameWon = thePlayer
        elif grid[x-1][y] == thePlayer and grid[x+1][y] == thePlayer and grid[x+2][y] == thePlayer:
            gameWon = thePlayer
        elif grid[x-2][y] == thePlayer and grid[x-1][y] == thePlayer and grid[x+1][y] == thePlayer:
            gameWon = thePlayer
        elif grid[x-3][y] == thePlayer and grid[x-2][y] == thePlayer and grid[x-1][y] == thePlayer:
            gameWon = thePlayer
    elif x == 4 and gameWon == 0:
        if grid[x-1][y] == thePlayer and grid[x-2][y] == thePlayer and grid[x-3][y] == thePlayer:
            gameWon = thePlayer
        elif grid[x-1][y] == thePlayer and grid[x-2][y] == thePlayer and grid[x+1][y] == thePlayer:
            gameWon = thePlayer
        elif grid[x-1][y] == thePlayer and grid[x+1][y] == thePlayer and grid[x+2][y] == thePlayer:
            gameWon = thePlayer
    elif x == 5 and gameWon == 0:
        if grid[x-1][y] == thePlayer and grid[x-2][y] == thePlayer and grid[x-3][y] == thePlayer:
            gameWon = thePlayer
        elif grid[x-1][y] == thePlayer and grid[x-2][y] == thePlayer and grid[x+1][y] == thePlayer:
            gameWon = thePlayer
    elif x == 6 and gameWon == 0:
        if grid[x-1][y] == thePlayer and grid[x-2][y] == thePlayer and grid[x-3][y] == thePlayer:
            gameWon = thePlayer
    #Diagonals
    elif x < 4 and gameWon == 0:
        if y < 3:
            if grid[x+1][y+1] == thePlayer and grid[x+2][y+2] == thePlayer and grid[x+3][y+3] == thePlayer:
                gameWon = thePlayer
        elif y >= 3:
            if grid[x+1][y-1] == thePlayer and grid[x+2][y-2] == thePlayer and grid[x+3][y-3] == thePlayer:            
                gameWon = thePlayer
    elif x > 3 and gameWon == 0:
        if y < 3:
            if grid[x-1][y+1] == thePlayer and grid[x-2][y+2] == thePlayer and grid[x-3][y+3] == thePlayer:
                gameWon = thePlayer
        elif y >= 3:
                gameWon = thePlayer
    
    return gameWon


#Pre-Loop Code
theGrid = resettheGrid(theGrid)
grid = np.copy(theGrid)
grid.shape = (1,42)
curState = grid
#model = buildModel(grid)
model = keras.models.load_model('C4Model.tflearn')


#AI trains against random opponent with a sliding epsilon
while True:
    
    #Check for player input
    gameWon = 0
    reward = 0
    if humanPlayer == True:
        notMoved = True
    
    #Allow pygame to close or it freaks out
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        if event.type == pygame.KEYDOWN:
            humanPlayer = True
            if event.key == pygame.K_LEFT:
                cursPos -= 1
                if cursPos < 0:
                    cursPos = 6
            elif event.key == pygame.K_RIGHT:
                cursPos += 1
                if cursPos > 6:
                    cursPos = 0
            elif event.key == pygame.K_RETURN:
                y = getValid(cursPos)
                if y > -1:
                    notMoved = False
                    x = cursPos
                    theGrid[x][y] = 0.5
                    gameWon = detectWin(theGrid,x,y,False)
        
    
    #Draw Board
    dis.fill((0,0,0))
    
    for x in range(theGridX):
        for y in range(theGridY):
            if theGrid[x][y] == 0:
                theCol = noCol
            elif theGrid[x][y] == 1:
                theCol = p1Col
            else:
                theCol = p2Col
            pygame.draw.circle(dis, theCol, (x * cellX + offset, y * cellY + offset),circleRad)
    pygame.draw.circle(dis, (0,0,255), (cursPos * cellX + offset, 400),circleRad/2)
    
    if gamesPlayed > 0:              
        message(str(AIWin / (gamesPlayed)), 100,400,(255,0,0))
        
                
    #Random Move for P1  
    humanPlayer = False
    if humanPlayer == False:
        y = -1
        if modelSaved == False:
            randomMove = 0
            while y == -1:
                randomMove = random.randrange(7)
                y = getValid(randomMove)
    
            x = randomMove
            theGrid[x][y] = 0.5
            gameWon = detectWin(theGrid,x,y,False)
        else:
            #AI Move for P1 using previous best model
            if builtModel == False:
                model2 = keras.models.load_model('C4Model.tflearn')
                print("Playing against model version ", modelVersion)
                builtModel = True
            
            if np.random.rand() < epsilon:
                action2 = random.randrange(7) 
                y = getValid(action2)
            else:    
                grid = np.copy(theGrid)
                grid.shape = (1,42)
                action2 = np.argmax(model2.predict(grid, verbose=0))
                y = getValid(action2)

            while y == -1:
                action2 = random.randrange(7)
                y = getValid(action2)
        
            x = action2
            theGrid[x][y] = 0.5
            gameWon = detectWin(theGrid,x,y,False)   
    

    
    #AI Move for P2
    if notMoved == False or humanPlayer == False:
        prevState = curState
        y = -1
        predictedMove = False
        action = 0
        
        if humanPlayer == True:
            epsilon = 0.1
            
        if np.random.rand() < epsilon:
            action = random.randrange(7) 
            y = getValid(action)
        else:
            predictedMove = True
            grid = np.copy(theGrid)
            grid.shape = (1,42)
            action = np.argmax(model.predict(grid, verbose=0))
            y = getValid(action)

        while y == -1:
            if predictedMove == True:
                reward = -.2
            action = random.randrange(7)
            y = getValid(action)

        x = action
        theGrid[x][y] = 1.0
        grid = np.copy(theGrid)
        grid.shape = (1,42)
        curState = grid
        if gameWon == 0:
            gameWon = detectWin(theGrid,x,y,False)
            #Check to see if move blocked opponent from winning
            if gameWon == 0 and detectWin(theGrid,x,y,True) > 0:
                reward = 0.3
        if gameWon > 0:
            isOver = True
            if gameWon == 1:
                reward = 1.0
            
        episode = [prevState, action, reward, curState, isOver]
        memory.append(episode)
        if (len(memory) > maxMemory):
            del memory[0]
        
        #GetData    
        envSize = prevState.size
        memSize = len(memory)
        dataSize = min(memSize, dataSize)
        inputs = np.zeros((dataSize, envSize))
        targets = np.zeros((dataSize, 7))
        for i, j in enumerate(np.random.choice(range(memSize), dataSize, replace=False)):
            envState, action, reward, envStateNext, gameOver = memory[j]
            inputs[i] = envState
            # There should be no target values for actions not taken.
            targets[i] = model.predict(envState, verbose=0)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(model.predict(envStateNext, verbose=0))
            if isOver == True:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + discount * Q_sa
            
                model.fit(
                    inputs,
                    targets,
                    epochs=32,
                    batch_size=32,
                    verbose=0,
                    )
                #If facing a human player, learn
        
        moves += 2
    pygame.display.update()
    clock.tick(gameSpeed)

    
    if moves >= (theGridX * theGridY)-1 or gameWon > 0:
        
        #Save model if facing a human
        if humanPlayer == True:
            model.save('C4Model.tflearn')
            
        #Detect Draw
        if gameWon == 0:
            drawGames += 1
        moves = 0
        isOver = False
        gamesPlayed += 1
        if epsilon > .1:
            epsilon -= epsilonDecay
        
        pygame.display.update()
        clock.tick(gameSpeed)
        
        if gameWon > 0:
            if gameWon == 1:
                AIWin += 1
            else:
                ranWin += 1
            if gamesPlayed == 199:
                print("AI: ", AIWin, "Rand: ", ranWin)
            pygame.display.update()
            clock.tick(gameSpeed)
        
        #Save/Load New AI after 200 games
        if gamesPlayed > totalGames:  
            
            #Plot Game
            setsPlayed += 1
            if setsPlayed > 1:
                if AIWin == 0:
                    AIWin = 1 #Prevent divide by 0
                winPercent = AIWin / (gamesPlayed - drawGames)
                roundPlot.append(setsPlayed)
                scoreRecord.append(winPercent)
                plt.plot(roundPlot,scoreRecord)
                plt.pause(0.05)
                print ("win Percent = ", winPercent, "%")                

            
            
            if modelSaved == False:
                print("New model saved!")
                modelSaved = True
                model.save('C4Model.tflearn')
            else:
                #if winPercent >= bestWinPercent + winImprove:
                if winPercent >= 0:    
                    bestWinPercent += winImprove
                    modelVersion += 1
                    print("New model ", modelVersion, " saved!")
                    model.save('C4Model.tflearn')
                else:
                    print("Loading model #", modelVersion)
                    model = keras.models.load_model('C4Model.tflearn')
            
            AIWin = 1
            ranWin = 0
            drawGames = 0
            gamesPlayed = 0
            winPercent = 0
            epsilon = startEpsilon
            #builtModel = False
        
        theGrid = resettheGrid(theGrid)
    