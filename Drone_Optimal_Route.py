# Author: Kasey French
# April 13th 2019
#
#

"""
You have just received a job in charge of the flight planning system for a drone delivery service. You're responsible for designing a system that when given a virtual map of an area can map out a flight path for the drone that will maximize the number of customers served.
Problem:

Write a function which takes as parameters

A 2D array representing the flight area. The value of a specific coordinate in the matrix represents the number of customers at that location.
A tuple representing the take off location of the drone
A tuple representing the landing location of the drone
This function should then return an integer representing the maximum customers that could be serviced on the optimal flight path.

Conditions:

A flight path cannot overlap itself.
Any point with a value of '-1' is a no-fly area. The drone cannot fly through it.
(Extra Credit) you can extend the function to include an integer "gas" parameter which indicates the maximum number of spaces the drone can traverse.
"""


"""
My solution to this problem involves tackling from the perspective of graph theory and creating
a mapping of nodes for each point and the map anc creating a relationships between the various
nodes. Each matrix element in the input data is represented as a node, and each node is
connected to it's adjacent nodes via the edge class. The Depth Search algorithm uses recursion
at each node to explore each path that could be taken at that step, weighted by the number
of customers at each node. The optimal path is defined for me as the path with the max value
of the customers served divided by the length of the path since this seemed to align with the
example's definition of optimization. The optional parameter of gas is also included.
"""
import numpy as np


# Basic data structure representing data and nodes with their relationships.
class DiGraph(object):
	def __init__(self, data):
		self.data = np.array(data)
		self.nodes = []
		self.edges = {}

	# Converts the input data to a list of node objects.
	def createNodes(self):
		(x,y) = self.data.shape
		x = range(0,x)
		y = range(0,y)
		for i in x:
			for j in y:
				self.nodes.append(Node(position = (i,j), weight = self.data[i][j]))

	# Creates edge relationship between adjacent nodes.
	def createEdges(self):
		for node1 in self.nodes:
			self.edges[node1] = []
			for node2 in self.nodes:
				dist = np.array(np.subtract(node1.position, node2.position))
				dist = np.linalg.norm(dist)
				if np.abs(dist) == 1:
					self.edges[node1].append(node2)

	# Returns the nodes directly adjacent to inut node.
	def childrenOf(self, node):
		return self.edges[node]

	def size(self):
		return self.data.shape[0] * self.data.shape[1]

	# Searches the graph for the node at the given tuple position.
	def findNode(self, pos):
		x,y = pos
		for node in self.nodes:
			if (x,y) == node.position:
				return node
		raise Exception("Error: Node not in map")


class Node(object):
	def __init__(self, position, weight):
		self.weight = weight
		self.position = position

	def __str__(self):
		return ("({},{})".format(self.position[0],self.position[1]))

class Edge(object):
	def __init__(self, start, end):
		self.start = start
		self.end = end

# Finds the "optimal path" to from start to end nodes on the input graph.
def DepthSearch(graph, start, end, path, optimalPath, optimalEfficiency, gas = None):
	path = path + [start]
	if start == end:
		return path
	if start.weight < 0 or end.weight < 0:
		raise Exception("Start or end point incorrect: Drone cannot fly there")

	if gas == None:
		maxLength = graph.size()
	else:
		maxLength = gas

	for node in graph.childrenOf(start):
		if node not in path and node.weight >= 0:
			newPath = DepthSearch(graph, node, end, path, optimalPath, optimalEfficiency, gas = maxLength)
			efficiency = 0
			if len(newPath) > 0 and len(newPath) <= maxLength:
				for node in newPath:
					efficiency += node.weight
				efficiency = efficiency / len(newPath)
				if efficiency > optimalEfficiency:
					optimalEfficiency = efficiency
					optimalPath = newPath

	return optimalPath

# Visualizes the path of the drone for a given path
def printPath(path):
	result = ''
	for i in range(len(path)):
		result = result + str(path[i])
		if i != len(path) - 1:
			result = result + '->'
	return result

# Takes in input data for map and returns the optimal number of customers a drone can visit on it
def optimalFlightPath(input_map, start, end, gas = None):
  graph = DiGraph(input_map)
  graph.createNodes()
  graph.createEdges()
  startNode = graph.findNode(start)
  endNode = graph.findNode(end)
  if gas != None:
  	gas = gas + 1
  path = DepthSearch(graph, startNode, endNode, path = [], optimalPath = [], optimalEfficiency = 0, gas = gas)
  #print(printPath(path))
  numCustomers = 0
  for node in path:
  	numCustomers += node.weight

  return numCustomers

def main():
	input_map = [
    [0, 0, 1, 1, 5],
    [1, 1, 1, -1, 7],
    [1, 1, 1, -1, 1],
    [1, 1, 1, -1, 9],
    [5, 1, 1, 1, 0],
	]

	# test_map = np.random.randint(-1, 2, size = (8,8))

	customers = optimalFlightPath(input_map, (0,0), (4,4), gas = None)
	print(customers)

if __name__ == "__main__":
	main()









