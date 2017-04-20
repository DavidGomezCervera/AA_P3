# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters

class NullGraphics:
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent:
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        "gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]"
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

# AGENTE DE APRENDIZAJE AUTOMATICO UC3M
import os
class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0

        #Obtenemos la direccion del fichero

        #SameMaps -----------------------------------------------
        path = os.getcwd() + "/Outputs/training_initial_v3_1A.arff"

        #Abrimos el fichero
        f = open(path,'a')

        statInfo = os.stat(path)
        
        if (statInfo.st_size == 0):
            s = "@RELATION pacman\n\n" \
                + "@ATTRIBUTE pacx numeric\n" \
                + "@ATTRIBUTE pacy numeric\n" \
                + "@ATTRIBUTE legal_north {true, false}\n" \
                + "@ATTRIBUTE legal_east {true, false}\n" \
                + "@ATTRIBUTE legal_south {true, false}\n" \
                + "@ATTRIBUTE legal_west {true, false}\n" \
                + "@ATTRIBUTE g1_x NUMERIC\n" \
                + "@ATTRIBUTE g1_y NUMERIC\n" \
                + "@ATTRIBUTE g2_x NUMERIC\n" \
                + "@ATTRIBUTE g2_y NUMERIC\n" \
                + "@ATTRIBUTE g3_x NUMERIC\n" \
                + "@ATTRIBUTE g3_y NUMERIC\n" \
                + "@ATTRIBUTE g4_x NUMERIC\n" \
                + "@ATTRIBUTE g4_y NUMERIC\n" \
                + "@ATTRIBUTE g1_dis NUMERIC\n" \
                + "@ATTRIBUTE g2_dis NUMERIC\n" \
                + "@ATTRIBUTE g3_dis NUMERIC\n" \
                + "@ATTRIBUTE g4_dis NUMERIC\n" \
                + "@ATTRIBUTE num_walls NUMERIC\n" \
                + "@ATTRIBUTE alive_ghosts NUMERIC\n" \
                + "@ATTRIBUTE score NUMERIC\n" \
                + "@ATTRIBUTE future_score NUMERIC\n" \
                + "@ATTRIBUTE future_alive_ghosts NUMERIC\n" \
                + "@ATTRIBUTE last_action {Stop, North, East, South, West}\n" \
                + "@ATTRIBUTE g1_relPos {-1,0,1,2,3,4,5,6,7,8}\n" \
                + "@ATTRIBUTE g2_relPos {-1,0,1,2,3,4,5,6,7,8}\n" \
                + "@ATTRIBUTE g3_relPos {-1,0,1,2,3,4,5,6,7,8}\n" \
                + "@ATTRIBUTE g4_relPos {-1,0,1,2,3,4,5,6,7,8}\n" \
                + "@ATTRIBUTE g1_closest {true, false}\n" \
                + "@ATTRIBUTE g2_closest {true, false}\n" \
                + "@ATTRIBUTE g3_closest {true, false}\n" \
                + "@ATTRIBUTE g4_closest {true, false}\n" \
                + "@ATTRIBUTE north_best {true, false}\n" \
                + "@ATTRIBUTE east_best {true, false}\n" \
                + "@ATTRIBUTE south_best {true, false}\n" \
                + "@ATTRIBUTE west_best {true, false}\n" \
                + "@ATTRIBUTE action {North, East, South, West}\n\n" \
                + "@DATA\n"

            s = "@RELATION pacman\n\n" \
                + "@ATTRIBUTE pacx numeric\n" \
                + "@ATTRIBUTE pacy numeric\n" \
                + "@ATTRIBUTE legal_north {true, false}\n" \
                + "@ATTRIBUTE legal_east {true, false}\n" \
                + "@ATTRIBUTE legal_south {true, false}\n" \
                + "@ATTRIBUTE legal_west {true, false}\n" \
                + "@ATTRIBUTE g1_x NUMERIC\n" \
                + "@ATTRIBUTE g1_y NUMERIC\n" \
                + "@ATTRIBUTE g1_dis NUMERIC\n" \
                + "@ATTRIBUTE num_walls NUMERIC\n" \
                + "@ATTRIBUTE alive_ghosts NUMERIC\n" \
                + "@ATTRIBUTE score NUMERIC\n" \
                + "@ATTRIBUTE future_score NUMERIC\n" \
                + "@ATTRIBUTE future_alive_ghosts NUMERIC\n" \
                + "@ATTRIBUTE last_action {Stop, North, East, South, West}\n" \
                + "@ATTRIBUTE g1_relPos {-1,0,1,2,3,4,5,6,7,8}\n" \
                + "@ATTRIBUTE g1_closest {true, false}\n" \
                + "@ATTRIBUTE north_best {true, false}\n" \
                + "@ATTRIBUTE east_best {true, false}\n" \
                + "@ATTRIBUTE south_best {true, false}\n" \
                + "@ATTRIBUTE west_best {true, false}\n" \
                + "@ATTRIBUTE action {North, East, South, West}\n\n" \
                + "@DATA\n"

            f.write(s)

        f.close()
        
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table


    def printLineData(self, gameState, move):

        #Obtenemos el path del fichero de datos que hemos definido.

        # SameMaps -----------------------------------------------
        path = os.getcwd() + "/Outputs/training_initial_v3_1A.arff"

        #Lo abrimos con el flag 'a' para que concatene el contenido al final del fichero, y asi no sobreescribirlo.
        dataFile = open(path, 'a')

        data = ""


        '''
        ATRIBUTOS BASICOS ----------------------------------------------------------------------------------------------
        '''


        #Obtenemos la posicion del pacman ------------------------------------------ 1, 2 ------------------------------
        data = data + str(gameState.data.agentStates[0].getPosition()[0]) + "," + str(gameState.data.agentStates[0].getPosition()[1]) + ","   

        #Obtenemos los movimientos legales, descartando el STOP. -------------------- 3, 4, 5, 6 -----------------------

        # Guardamos true para aquellas acciones que sean legales y false para aquellas que no (siempre en el mismo orden)
        actions = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]
        legal = gameState.getLegalPacmanActions()

        for action in actions:
            if action in legal:
                data = data + "true,"
            else:
                data = data + "false,"

        #Obtenemos la posicion del pacman (x,y)
        pos_pac = gameState.data.agentStates[0].getPosition()

        #Obtenemos las posiciones de los fantasmas ---------------------------- 7, 8, 9, 10, 11, 12, 13, 14 ------------
        for i in range(1, gameState.getNumAgents()):

            data = data + str(gameState.data.agentStates[i].getPosition()[0]) + "," + str(gameState.data.agentStates[i].getPosition()[1]) + ","


        #Obtenemos las distancias a los fantasmas ------------------------------ 15, 16, 17, 18 ------------------------
        for i in range(1, gameState.getNumAgents()):

            #Calculmos la distancia real (mazedistance) al fantasma i
            pos_ghost = gameState.data.agentStates[i].getPosition()

            distance = self.distancer.getDistance(pos_pac, pos_ghost)

            #Si la distancia es mayor a 1000 significa que el fantasma en cuestion ya ha sido comido
            if (distance > 1000):
                data = data + ("-1,")
            else:
                data = data + str(distance) + ","


        #Obtenemos el numero de muros ---------------------------------------- 19 --------------------------------------
        num_walls = 0

        for i in range (pos_pac[0]-1, pos_pac[0]+1):

            if i < 0 or i >= gameState.data.layout.width:
                continue

            for j in range (pos_pac[1]-1, pos_pac[1]+1):

                if j < 0 or j >= gameState.data.layout.height:
                    continue

                if gameState.getWalls()[i][j] == "%":
                    num_walls += 1

        data = data + str(num_walls) + ","

        #Obtenemos el numero de fantasmas vivos en este tick ----------------- 20 ---------------------------------------
        alive_ghosts = 0
        for i in gameState.getLivingGhosts():
            if (i == True):
                alive_ghosts += 1

        data = data + str(alive_ghosts) + ","

        #Obtenemos la puntuacion actual --------------------------------------- 21 -------------------------------------
        data = data + str(gameState.getScore()) + ","

        #Obtenemos la puntuacion en el tick siguiente -------------------------- 22 ------------------------------------
        future_score = gameState.getScore() - 1

        current_min = 1000000
        for i in range(1, gameState.getNumAgents()):

            # Calculmos la distancia manhattan al fantasma i.
            pos_ghost = gameState.data.agentStates[i].getPosition()

            distance = self.distancer.getDistance(pos_pac, pos_ghost)

            #Vamos almacenando el fantasma mas cercano
            if (distance < current_min):
                current_min = distance

        #Si hay algun fantasma a distancia 1, la puntuacion aumentara en 100
        if (current_min == 1):
            future_score = future_score + 100

        #Guardamos la puntuacion futura ----------------------------------- 23 -----------------------------------------
        data = data + str(future_score) + ","

        #Obtenemos el numero de fantasmas vivos en el siguiente tick
        if (current_min == 1):
            data = data + str(alive_ghosts - 1) + ","
        else:
            data = data + str(alive_ghosts) + ","

        #Obtenemos el movimiento anterior ---------------------------------- 24 ----------------------------------------
        data = data + str(gameState.data.agentStates[0].getDirection()) + ","



        '''
        MEJORAS ATRIBUTOS -----------------------------------------------------------------------------------------------
        '''

        # Obtenemos las posiciones relativas de los fantasmas con respecto del pacman ----- 25, 26, 27, 28 -------------
        for i in range(1, gameState.getNumAgents()):

            pos_ghost = gameState.data.agentStates[i].getPosition()

            if (pos_ghost[1] < 3):
                data = data + "-1,"
                continue

            # Si el fantasma esta en la misma posicion lo indicamos como 0
            if (pos_ghost == pos_pac):
                data = data + "0,"

            # Determinamos las posiciones relativas
            # {NORTH = 1, NORTH_EAST = 2, EAST = 3, SOUTH_EAST = 4, SOUTH = 5, SOUTH_WEST = 6, WEST = 7, NORTH_WEST = 8}.
            if (pos_ghost[0] > pos_pac[0]):
                if (pos_ghost[1] > pos_pac[1]):
                    data = data + "2,"
                elif (pos_ghost[1] < pos_pac[1]):
                    data = data + "4,"
                else:
                    data = data + "3,"
            elif (pos_ghost[0] < pos_pac[0]):
                if (pos_ghost[1] > pos_pac[1]):
                    data = data + "8,"
                elif (pos_ghost[1] < pos_pac[1]):
                    data = data + "6,"
                else:
                    data = data + "7,"
            else:
                if (pos_ghost[1] > pos_pac[1]):
                    data = data + "1,"
                else:
                    data = data + "5,"


        #Obtenemos el fantasma mas cercano. ----------------------------------------------------------------------------
        index = -1
        current_min = 10000000

        for i in range(1, gameState.getNumAgents()):

            # Calculmos la distancia manhattan al fantasma i.
            ghostPos = gameState.data.agentStates[i].getPosition()

            distance = self.distancer.getDistance(pos_pac, ghostPos)

            # Nos vamos quedando con el fantasma mas cercano.
            if (distance < current_min):
                current_min = distance
                index = i

        for i in range(1, gameState.getNumAgents()):
            if (i == index):
                data = data + "true,"
            else:
                data = data + "false,"

        #Encontramos la mejor accion -----------------------------------------------------------------------------------

        newMinDistance = 100000000

        best_action = Directions.NORTH

        # Simulamos cada accion posible y nos quedamos con aquella que nos acerque mas
        for action in legal:

            # Calculamos la distancia (MazeDistance) al fantasma mas cercano tras hacer una accion.
            newPos = Actions.getSuccessor(pos_pac, action)

            newDistance = self.distancer.getDistance(newPos, gameState.data.agentStates[index].getPosition())

            # Vamos almacenando la accion que mas nos acerque.
            if (newDistance < newMinDistance):
                best_action = action
                newMinDistance = newDistance

        for action in actions:
            if (action == best_action):
                data = data + "true,"
            else:
                data = data + "false,"

        '''
        ---------------------------------------------------------------------------------------------------------------
        '''



        '''
        ESCRITURA EN FICHERO -------------------------------------------------------------------------------------------
        '''

        #Obtenemos el movimiento que hemos realizado este turno --------------------------------------------------------
        data = data + move

        #Escrimos en el fichero la nueva linea -------------------------------------------------------------------------
        dataFile.write(data + "\n")

        #Cerramos el fichero.
        dataFile.close()

        return data    


    def chooseAction(self, gameState):
        
        self.countActions = self.countActions + 1

        #Obtenemos la posicion del pacman (x, y)
        pacmanPos = gameState.data.agentStates[0].getPosition()    

        #Movimiento por defecto        
        move = Directions.STOP

        legal = gameState.getLegalActions(0) ##Legal position from the pacman

        dots = False
        #Se da prioridad a encontrar las pildoras de comida.
        if (dots):#gameState.getNumFood() > 0):

            minDotDistance = 10000000

            dotPos = (0,0)

            #Encontramos la posicion de la comida mas cercana al pacman
            i = 0
            for width in gameState.data.food:
                j = 0
                for height in width:

                    #Si la posicion es true, hay una pildora aqui
                    if(height == True):
                        aux = (i,j)
                        dotDistance = self.distancer.getDistance(pacmanPos, aux)

                        if(dotDistance < minDotDistance):
                            minDotDistance = dotDistance
                            dotPos = (i,j)
                    j += 1
                i += 1

            newMinDistance = 1000000    

             #Simulamos cada accion posible y nos quedamos con aquella que nos acerque mas        
            for action in legal:

                #Calculamos la distancia (MazeDistance) al fantasma mas cercano tras hacer una accion.
                newPos = Actions.getSuccessor(pacmanPos, action)
            
                newDistance = self.distancer.getDistance(newPos, dotPos)

                #Vamos almacenando la accion que mas nos acerque.
                if (newDistance < newMinDistance):
                    move = action
                    newMinDistance = newDistance        

        else:
            
            index = -1
            current_min = 10000000

            #Encontramos el fantasma mas cercano.
            for i in range(1, gameState.getNumAgents()):

                #Calculmos la distancia manhattan al fantasma i.
                ghostPos = gameState.data.agentStates[i].getPosition()

                distance = self.distancer.getDistance(pacmanPos, ghostPos)

                #Nos vamos quedando con el fantasma mas cercano.
                if (distance < current_min):
                    current_min = distance
                    index = i

            newMinDistance = 10000000

            #Simulamos cada accion posible y nos quedamos con aquella que nos acerque mas
            for action in legal:

                #Calculamos la distancia (MazeDistance) al fantasma mas cercano tras hacer una accion.
                newPos = Actions.getSuccessor(pacmanPos, action)

                newDistance = self.distancer.getDistance(newPos, gameState.data.agentStates[index].getPosition())

                #Vamos almacenando la accion que mas nos acerque.
                if (newDistance < newMinDistance):
                    move = action
                    newMinDistance = newDistance
           
        self.printLineData(gameState, move)
    
        return move

#Importamos la maaquina virtual de Java.
import weka.core.jvm as jvm
#Importamos los algoritmos de clustering.
from weka.clusterers import Clusterer
#Importamos los cargadores de arff.
from weka.core.converters import Loader


class ClusterAgent (BustersAgent):


    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

        #Definimos si se usa la distancia (true para v1 y v2, false para v3)
        self.dis = True

        #Para calcular los valores de la clase en las politicas.
        self.clusters = 8
        self.classes = 4
        self.classCounts = [[0 for i in range(self.classes)]for j in range(self.clusters)]

        self.classIndex = 2
        self.clusterIndex = 3

        self.readInstances()

        #Esto nos servira para guardar las instancias de entrenamiento.
        self.numInstances = 52
        self.numAttributes = 4
        #self.instances = [[" " for i in range(self.numAttributes)] for j in range(self.numInstances)]
        self.ins = [" " for i in range(self.numInstances)]

        #Para usar la libreria debemos usar la maquina virtual de java, JVM
        jvm.start()

        #Creamos el modelo
        loader = Loader(classname="weka.core.converters.ArffLoader")
        data = loader.load_file("/home/dot/Escritorio/Universidad/Machine Learning/practica 2/Outputs/agent_header.arff")

        self.clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", str(self.clusters)])
        self.clusterer.build_clusterer(data)

        print(self.clusterer)

        #Aplicamos la politica
        self.politicaMax()


    def readInstances(self):

        #Direccion del fichero agente (instancias sin cabecera).
        path = os.getcwd() + "/Outputs/agent.arff"

        f = open(path, 'r')

        index = 0

        #Leemos cacda instancia
        for line in f:

            #Obtenemos los valores de los atributos (String)
            values = line.split(",")

            #Obtenemos el valor de la clase, de Norte a Oeste (0 - 3)
            classValue = 0
            classAtt = values[self.classIndex]
            if (classAtt == "East"):
                classValue = 1
            elif (classAtt == "South"):
                classValue = 2
            elif (classAtt == "West"):
                classValue = 3

            #Obtenemos el valor del cluster.
            cluster = values[self.clusterIndex]

            #Incrementamos la cuenta de la clase para el cluster.
            self.classCounts[int(cluster[-2:]) - 1][classValue] += 1

        f.close()

    #Calcula la clase mayoritaria para cada cluster
    def politicaMax(self):

        self.max = [0 for i in range(self.clusters)]

        for i in range(self.clusters):

            temp_max = 0
            class_index = 0

            for j in range(self.classes):

                if (self.classCounts[i][j] > temp_max):

                    temp_max = self.classCounts[i][j]
                    class_index = j

            self.max[i] = class_index
            #print(class_index)

        '''
        for i in range(self.clusters):
            print(self.max[i])
        '''

    def chooseAction(self, gameState):

        path = os.getcwd() + "/Outputs/newInstance.arff"

        f = open(path, 'w')

        if (self.dis):
            data = "@RELATION pacman\n" \
                    + "@ATTRIBUTE dis NUMERIC\n" \
                    + "@ATTRIBUTE relPos {-1,0,1,2,3,4,5,6,7,8}\n\n" \
                    + "@DATA\n"
        else:
            data = "@RELATION pacman\n" \
                   + "@ATTRIBUTE relPos {-1,0,1,2,3,4,5,6,7,8}\n\n" \
                   + "@DATA\n"


        # Obtenemos la posicion del pacman (x,y)
        pos_pac = gameState.data.agentStates[0].getPosition()


        # Obtenemos las distancias a los fantasmas
        for i in range(1, gameState.getNumAgents()):

            # Calculmos la distancia real (mazedistance) al fantasma i
            pos_ghost = gameState.data.agentStates[i].getPosition()

            distance = self.distancer.getDistance(pos_pac, pos_ghost)

            #Normalizacion: (distance - min)/(max - min): min = 1, max = 21
            distance = (distance - 1) / (21 - 1)

            # Si la distancia es mayor a 1000 significa que el fantasma en cuestion ya ha sido comido
            if (self.dis):
                if (distance > 1000):
                    data = data + ("-1,")
                else:
                    data = data + str(distance) + ","


        # Obtenemos las posiciones relativas de los fantasmas con respecto del pacman
        for i in range(1, gameState.getNumAgents()):

            pos_ghost = gameState.data.agentStates[i].getPosition()

            if (pos_ghost[1] < 3):
                data = data + "-1,"
                continue

            # Si el fantasma esta en la misma posicion lo indicamos como 0
            if (pos_ghost == pos_pac):
                data = data + "0,"

            # Determinamos las posiciones relativas
            # {NORTH = 1, NORTH_EAST = 2, EAST = 3, SOUTH_EAST = 4, SOUTH = 5, SOUTH_WEST = 6, WEST = 7, NORTH_WEST = 8}.
            if (pos_ghost[0] > pos_pac[0]):
                if (pos_ghost[1] > pos_pac[1]):
                    data = data + "2,"
                elif (pos_ghost[1] < pos_pac[1]):
                    data = data + "4,"
                else:
                    data = data + "3,"
            elif (pos_ghost[0] < pos_pac[0]):
                if (pos_ghost[1] > pos_pac[1]):
                    data = data + "8,"
                elif (pos_ghost[1] < pos_pac[1]):
                    data = data + "6,"
                else:
                    data = data + "7,"
            else:
                if (pos_ghost[1] > pos_pac[1]):
                    data = data + "1,"
                else:
                    data = data + "5,"

        data = data + "\n"

        #print(data)

        f.write(data)

        f.close()

        loader = Loader(classname="weka.core.converters.ArffLoader")
        newData = loader.load_file("/home/dot/Escritorio/Universidad/Machine Learning/practica 2/Outputs/newInstance.arff")

        dir = 4
        direction = Directions.STOP

        for inst in newData:
            cl = self.clusterer.cluster_instance(inst)
            #print(cl)
            dir = self.max[cl]
            #print(dir)


        if (dir == 0):
            direction = Directions.NORTH
        elif (dir == 1):
            direction = Directions.EAST
        elif (dir == 2):
            direction = Directions.SOUTH
        elif (dir == 3):
            direction = Directions.WEST

        #print(direction)
        return direction
        #return Directions.STOP
