# -*- coding: utf-8 -*-
"""
SER501 Assignment 3 scaffolding code
created by: Xiangyu Guo
"""
import sys

# =============================================================================


class Graph(object):
    """docstring for Graph"""
    user_defined_vertices = []
    dfs_timer = 0

    def __init__(self, vertices, edges):
        super(Graph, self).__init__()
        n = len(vertices)
        self.matrix = [[0 for x in range(n)] for y in range(n)]
        self.vertices = vertices
        self.edges = edges
        for edge in edges:
            x = vertices.index(edge[0])
            y = vertices.index(edge[1])
            self.matrix[x][y] = edge[2]

    def display(self):
        print(self.vertices)
        for i, v in enumerate(self.vertices):
            print(v, self.matrix[i])

    def transpose(self):
        n = len(self.matrix)
        t_matrix = [[0 for x in range(n)] for y in range(n)]
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                t_matrix[j][i] = self.matrix[i][j]
        self.matrix = t_matrix

    def in_degree(self):
        print("In degree of the graph:")
        n = len(self.vertices)
        degree = [0] * n
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                if(self.matrix[j][i] > 0):
                    degree[i] = degree[i] + 1
        self.print_degree(degree)

    def out_degree(self):
        print("Out degree of the graph:")
        # ToDo
        n = len(self.vertices)
        degree = [0] * n
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                if(self.matrix[i][j] > 0):
                    degree[i] = degree[i] + 1
        self.print_degree(degree)

    def dfs_traverse(self, start):
        neighs = [i for i, x in enumerate(self.matrix[start]) if x == 1 and i not in self.visited]
        for n in neighs:
            if n not in self.visited:
                self.count += 1
                self.visited.append(n)
                if self.first_discovered[n] == sys.maxsize:
                    self.first_discovered[n] = self.count
                self.dfs_traverse(n)
                self.count += 1
                self.end_discovered[n] = self.count
        if len(neighs) == 0:
            self.end_discovered[start] = self.count

    def dfs_on_graph(self):
        self.visited = []
        self.first_discovered = [sys.maxsize] * len(self.vertices)
        self.end_discovered = [sys.maxsize] * len(self.vertices)
        self.count = 0
        for start in range(len(self.vertices)):
            if start not in self.visited:
                self.count += 1
                self.first_discovered[start] = self.count
                self.visited.append(start)
                self.dfs_traverse(start)
                self.count += 1
                self.end_discovered[start] = self.count
        self.print_discover_and_finish_time(self.first_discovered, self.end_discovered)

    def prim(self, root):
        print("Prim:")
        d = [sys.maxsize] * len(self.vertices)
        pi = [None] * len(self.vertices)
        d[self.vertices.index(root)] = 0
        mst = [False] * len(self.vertices)
        pi[self.vertices.index(root)] = None
        self.print_d_and_pi("Initial", d, pi)
        for i in range(len(self.vertices)):
            u = self.minDistance(d, mst)
            mst[u] = True
            for v in range(len(self.vertices)):
                if self.matrix[u][v] > 0 and mst[v] == False and d[v] > self.matrix[u][v]:
                    d[v] = self.matrix[u][v]
                    pi[v] = self.vertices[u]
            self.print_d_and_pi(i, d, pi)

    def minDistance(self, distance, mst):
        min_distance = sys.maxsize
        for v in range(len(self.vertices)):
            if distance[v] < min_distance and mst[v] == False:
                min_distance = distance[v]
                closest_node = v
        return closest_node

    def bellman_ford(self, source):
        print("Bellman Ford")
        d = [sys.maxsize] * len(self.vertices)
        d[self.vertices.index(source)] = 0
        pi = [None] * len(self.vertices)
        self.print_d_and_pi("Initial", d, pi)
        for i in range(len(self.vertices)-1):
            for edge in self.edges:
                if d[self.vertices.index(edge[0])] != sys.maxsize and d[self.vertices.index(edge[1])] > d[self.vertices.index(edge[0])]+edge[2]:
                    d[self.vertices.index(edge[1])] = d[self.vertices.index(edge[0])]+edge[2]
                    pi[self.vertices.index(edge[1])] = edge[0]
            self.print_d_and_pi(i, d, pi)
        for edge in self.edges:
            if d[self.vertices.index(edge[1])] > d[self.vertices.index(edge[0])]+edge[2]:
                print("No Solution")
                return

    def dijkstra(self, source):
        print("Djikstra : ")
        n = len(self.vertices)
        d = [sys.maxint] * n
        d[self.vertices.index(source)] = 0
        pi = [None] * len(self.vertices)
        visited = [False] * n
        pi[self.vertices.index(source)] = None
        self.print_d_and_pi("Initial", d, pi)
        for i in range(n):
            u = self.minDistance(d, visited)
            visited[u] = True
            for v in range(n):
                if self.matrix[u][v] > 0 and visited[v] == False and d[v] > d[u] + self.matrix[u][v]:
                    d[v] = d[u] + self.matrix[u][v]
                    pi[v] = self.vertices[u]
            self.print_d_and_pi(i, d, pi)

    def print_d_and_pi(self, iteration, d, pi):
        assert((len(d) == len(self.vertices)) and
               (len(pi) == len(self.vertices)))

        print("Iteration: {0}".format(iteration))
        for i, v in enumerate(self.vertices):
            print("Vertex: {0}\td: {1}\tpi: {2}".format(v, 'inf' if d[i] == sys.maxsize else d[i], pi[i]))

    def print_discover_and_finish_time(self, discover, finish):
        assert((len(discover) == len(self.vertices)) and
               (len(finish) == len(self.vertices)))
        for i, v in enumerate(self.vertices):
            print("Vertex: {0}\tDiscovered: {1}\tFinished: {2}".format(
                    v, discover[i], finish[i]))

    def print_degree(self, degree):
        assert((len(degree) == len(self.vertices)))
        for i, v in enumerate(self.vertices):
            print("Vertex: {0}\tDegree: {1}".format(v, degree[i]))


def main():
    # Thoroughly test your program and produce useful output.
    # Q1 and Q2
    graph = Graph(['1', '2'], [('1', '2',  1)])
    graph.display()
    graph.transpose()
    graph.display()
    graph.transpose()
    graph.display()
    graph.in_degree()
    graph.out_degree()
    graph.print_d_and_pi(1, [1, sys.maxsize], [2, None])
    graph.print_degree([1, 0])
    graph.print_discover_and_finish_time([0, 2], [1, 3])

    # Q3
    graph = Graph(['q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
                  [('q', 's', 1),
                   ('s', 'v', 1),
                   ('v', 'w', 1),
                   ('w', 's', 1),
                   ('q', 'w', 1),
                   ('q', 't', 1),
                   ('t', 'x', 1),
                   ('x', 'z', 1),
                   ('z', 'x', 1),
                   ('t', 'y', 1),
                   ('y', 'q', 1),
                   ('r', 'y', 1),
                   ('r', 'u', 1),
                   ('u', 'y', 1)])
    graph.display()
    graph.dfs_on_graph()

    # Q4 - Prim
    graph = Graph(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                  [('A', 'H', 6),
                   ('H', 'A', 6),
                   ('A', 'B', 4),
                   ('B', 'A', 4),
                   ('B', 'H', 5),
                   ('H', 'B', 5),
                   ('B', 'C', 9),
                   ('C', 'B', 9),
                   ('G', 'H', 14),
                   ('H', 'G', 14),
                   ('F', 'H', 10),
                   ('H', 'F', 10),
                   ('B', 'E', 2),
                   ('E', 'B', 2),
                   ('G', 'F', 3),
                   ('F', 'G', 3),
                   ('E', 'F', 8),
                   ('F', 'E', 8),
                   ('D', 'E', 15),
                   ('E', 'D', 15)])
    graph.prim('G')

    # Q5
    graph = Graph(['s', 't', 'x', 'y', 'z'],
                  [('t', 'x', 5),
                   ('t', 'y', 8),
                   ('t', 'z', -4),
                   ('x', 't', -2),
                   ('y', 'x', -3),
                   ('y', 'z', 9),
                   ('z', 'x', 7),
                   ('z', 's', 2),
                   ('s', 't', 6),
                   ('s', 'y', 7)])
    graph.bellman_ford('z')

    # Q5 alternate
    graph = Graph(['s', 't', 'x', 'y', 'z'],
                  [('t', 'x', 5),
                   ('t', 'y', 8),
                   ('t', 'z', -4),
                   ('x', 't', -2),
                   ('y', 'x', -3),
                   ('y', 'z', 9),
                   ('z', 'x', 4),
                   ('z', 's', 2),
                   ('s', 't', 6),
                   ('s', 'y', 7)])
    graph.bellman_ford('s')

    # Q6
    graph = Graph(['s', 't', 'x', 'y', 'z'],
                  [('s', 't', 3),
                   ('s', 'y', 5),
                   ('t', 'x', 6),
                   ('t', 'y', 2),
                   ('x', 'z', 2),
                   ('y', 't', 1),
                   ('y', 'x', 4),
                   ('y', 'z', 6),
                   ('z', 's', 3),
                   ('z', 'x', 7)])
    graph.dijkstra('s')


if __name__ == '__main__':
    main()
