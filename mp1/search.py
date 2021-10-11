# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #rows = maze.size.y
    #columns = maze.size.x
    q = []
    visited = []
    q.append([maze.start])
    visited.append([maze.start])
    target = maze.waypoints[0]
    while q:
        path = q.pop(0)
        cell = path[-1]
        if cell == target:
            return path 
        for neighborCell in maze.neighbors(cell[0], cell[1]):
            if neighborCell not in visited:
                visited.append(neighborCell)
                newPath = list(path)
                newPath.append(neighborCell)
                q.append(newPath)
    return []

import heapq

def findPath(maze, pointStart, pointEnd):
    """
    using astar with heuristic function to find min path between start and end

    @param maze: The maze to execute the search on.
    @param pointStart: the start point
    @param pointEnd: the target point

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    def heuristic(c):
        #helper function for manhattan distance
        return abs(c[0] - pointEnd[0]) + abs(c[1] - pointEnd[1])

    q = []  #store tuple ( g(n) + h(n), list of path )
    heapq.heapify(q)
    dic = {}    #dictionary store cell -> path gone so far
    heapq.heappush(q, (heuristic(pointStart), [pointStart]))
    dic[pointStart] = 0
    res = []
    
    while q:
        g_plus_h, cur_path = heapq.heappop(q)
        cell = cur_path[-1]
        if len(res) != 0 and g_plus_h >= len(res) - 1: continue
        if cell == pointEnd:
            res = cur_path
            continue 
        for neighborCell in maze.neighbors(cell[0], cell[1]):
            if dic.get(neighborCell) == None or dic.get(neighborCell) > len(cur_path):
                dic[neighborCell] = len(cur_path)
                newPath = list(cur_path)
                newPath.append(neighborCell)
                heapq.heappush(q, (len(cur_path) + heuristic(neighborCell), newPath))
    return res

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return findPath(maze, maze.start, maze.waypoints[0])

def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """

    toReturn = []
    waypoints = maze.waypoints
    adj = []
    for i in range(len(waypoints)):
        adj.append([])
        for j in range(len(waypoints)):
            if i == j:
                adj[i].append([])
            elif i < j:
                curPath = findPath(maze, waypoints[i], waypoints[j])
                adj[i].append(curPath)
            else:
                adj[i].append(adj[j][i][::-1])

    startToFirst = []
    for i in range(len(waypoints)):
        curPath = findPath(maze, maze.start, waypoints[i])
        startToFirst.append(curPath)


    def dfs(nums, permute):
        if not nums:
            curPathLen = len(startToFirst[permute[0]])
            for i in range(len(waypoints) - 1):
                curPathLen += len(adj[permute[i]][permute[i+1]]) - 1
            return curPathLen, permute

        min_len = -1
        min_permute = []
        for i in range(len(nums)):
            res_len, res_permute = dfs(nums[:i]+nums[i+1:], permute+[nums[i]])
            if min_len == -1 or res_len < min_len:
                min_len = res_len
                min_permute = res_permute
        return min_len, min_permute
    
    base_permute = [i for i in range(len(waypoints))]
    best_permute = dfs(base_permute, [])[1]
    toReturn = startToFirst[best_permute[0]]
    for i in range(len(best_permute) - 1):
        toReturn = toReturn + adj[best_permute[i]][best_permute[i+1]][1:]

    return toReturn
    
def h4(visited, idx, adj, len_path):
    if (len(visited) == len(adj[idx])): 
        return len_path

    min_j = []  #store lists on min index j
    len_min_path = []
    for j in range(len(adj[idx])):
        if idx != j and j not in visited:
            if len(len_min_path) == 0 or len(adj[idx][j]) - 1 < len_min_path[0]:
                min_j = [j]
                len_min_path = [len(adj[idx][j]) - 1]
            elif len(adj[idx][j]) - 1 == len_min_path[0]:
                min_j.append(j)
                len_min_path.append(len(adj[idx][j]) - 1)
    res = -1
    for num in range(len(min_j)):   #try edge with same min path
        ans = h4(visited + [min_j[num]], min_j[num], adj, len_path + len_min_path[num])
        if res == -1 or ans < res:
            res = ans
    return res

def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    waypoints = maze.waypoints
    adj = []
    for i in range(len(waypoints)):
        adj.append([])
        for j in range(len(waypoints)):
            if i == j:
                adj[i].append([])
            elif i < j:
                curPath = findPath(maze, waypoints[i], waypoints[j])
                adj[i].append(curPath)
            else:
                adj[i].append(adj[j][i][::-1])

    startToFirst = []
    for i in range(len(waypoints)):
        curPath = findPath(maze, maze.start, waypoints[i])
        startToFirst.append(curPath)


    def dfs(nums, permute, adj, curPath):
        if not nums:
            return curPath

        min_path = []
        for idx in range(len(nums)):
            path = []
            if not permute:
                path = startToFirst[nums[idx]]
            else:
                path = adj[permute[-1]][nums[idx]][1:]
            if len(min_path) != 0 and len(path) + h4(permute+[nums[idx]], nums[idx], adj, 0) - 3 >= len(min_path) - len(curPath):
                continue
            resPath = dfs(nums[:idx]+nums[idx+1:], permute+[nums[idx]], adj, curPath + path)
            if not min_path or len(resPath) < len(min_path):
                min_path = resPath
        return min_path
    
    base_permute = [i for i in range(len(waypoints))]
    toReturn = dfs(base_permute, [], adj, [])

    return toReturn

def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
    
            
