import logging
import math
import time
from collections import deque
from copy import deepcopy

import mrcfile
import numpy as np

from .pdb_reader_writer import PDB_Reader_Writer

def build_graph(pdb_file, ca_image, origin, remove_tail_loops):
    """Build graph and then clean it to improve final output"""
    graph = make_graph(pdb_file)
    graph.edge_check()
    graph.remove_side_chains()
    graph.remove_loops(ca_image, origin)
    graph.remove_single_ends()
    graph.remove_empty_nodes()
    graph.remove_side_chains()
    graph.remove_loops(ca_image, origin)
    graph.remove_single_ends()
    if remove_tail_loops:
        graph.remove_tail_loops()
    graph.remove_empty_nodes()
    return graph

class Node:
    """Simple Node class defining the location of a Ca atom along with which Ca
    atoms may be connected to it. There may be any number of connected Ca atoms."""

    def __init__(self, location):
        self.location = location
        self.edges = list()

    def get_location(self):
        return self.location

    def add_edge(self, edge):
        self.edges.append(edge)

    def get_num_edges(self):
        return len(self.edges)

    def get_edges(self):
        return self.edges



class Graph:
    """Graph class that defines the full backbone-Ca structure of the protein
    This class contains functions used to clean and improve the predicted
    backbone structure as a final step of post-processing"""

    def __init__(self):
        self.nodes = list()

    def add_node(self, node):
        self.nodes.append(node)

    def contains_location(self, location):
        for node in self.nodes:
            if node.get_location() == location:
                return True
        return False

    def get_node(self, location):
        for node in self.nodes:
            if node.get_location() == location:
                return node

        raise ValueError('No node found with given location')

    def edge_check(self):
        for node in self.nodes:
            index = 0
            while index < node.get_num_edges():
                if node.get_edges()[index] == node.get_location():
                    node.get_edges().remove(node.get_edges()[index])
                    index -= 1
                index += 1
            for edge in node.get_edges():
                if edge == node.get_location():
                    print("Edge Failure!")

    def get_end_nodes(self):
        end_nodes = list()
        for node in self.nodes:
            if node.get_num_edges() <= 1:
                end_nodes.append(node)
        return end_nodes

    def remove_single_ends(self):
        """This method removes all nodes/edges that are connected to a node
        that have 3 or more edges. It also removes Nodes that are connected
        to other end Nodes (a pair of disconnected Ca atoms)."""
        for node in self.nodes:
            if node.get_num_edges() == 1:  # This is a single connection
                neighbor_node = self.get_node(node.get_edges()[0])
                if neighbor_node.get_num_edges() >= 3 or neighbor_node.get_num_edges() == 1:
                    neighbor_node.edges.remove(node.get_location())
                    node.edges.remove(neighbor_node.get_location())
                else:
                    for node_2 in neighbor_node.get_edges():
                        next_node = self.get_node(node_2)
                        if next_node.get_num_edges() >= 3:
                            next_node.edges.remove(neighbor_node.get_location())
                            neighbor_node.edges.remove(next_node.get_location())
                            neighbor_node.edges.remove(node.get_location())
                            node.edges.remove(neighbor_node.get_location())

    def walk_until_trinary(self, location, visited):
        """A recursive method used to find the end of a trace
        An end of a trace is defined as the point where the trace ends or when
        the trace hits a Ca atom with 3 or more edges"""
        node = self.get_node(location)
        visited.append(location)
        if node.get_num_edges() == 2:
            for edge in node.get_edges():
                if edge not in visited:
                    visited = self.walk_until_trinary(edge, deepcopy(visited))
        return visited

    def walk_graph(self, location, visited, depth):
        """A recursive method similar to walk_until_trinary that finds the
        depth of trace. Also stops when it reaches a node that has 3 edges
        or a terminating node."""
        node = self.get_node(location)
        if node.get_num_edges() <= 2:
            visited.append(location)
            for edge in node.get_edges():
                if edge not in visited:
                    depth = self.walk_graph(edge, deepcopy(visited), depth + 1)
        return depth

    def remove_single_loop(self, ca_list):
        for index, ca in enumerate(ca_list):
            if index > 0:
                to_be_removed = self.get_node(ca)
                for edge in to_be_removed.edges:
                    self.get_node(edge).edges.remove(to_be_removed.location)
                to_be_removed.edges.clear()

    def remove_tail_loops(self):
        for node in self.nodes:
            if node.get_num_edges() >= 3:  # This is a trinary connection
                walk_lists = list()
                visited = list()
                visited.append(node.get_location())
                for edge in node.get_edges():
                    walk_lists.append(self.walk_until_trinary(edge, deepcopy(visited)))
                done = False
                for list1 in walk_lists:
                    if done:
                        break
                    for list2 in walk_lists:
                        if done:
                            break
                        if list1 != list2 and list1[len(list1) - 1] == list2[1] and len(list1) <= 4:
                            self.remove_single_loop(list1)
                            done = True

    def remove_loops(self, input_image, origin):
        """This method removes one side of a loop in the graph (a cycle). The
        side of the loop with the least density will be removed."""
        for node in self.nodes:
            if node.get_num_edges() >= 3:  # This is a trinary connection
                walk_lists = list()
                visited = list()
                visited.append(node.get_location())
                for edge in node.get_edges():
                    walk_lists.append(self.walk_until_trinary(edge, deepcopy(visited)))

                done = False
                for list1 in walk_lists:
                    if done:
                        break
                    for list2 in walk_lists:
                        if done:
                            break

                        # This removed a loop along the backbone that ends in a different trinary node.
                        if list1 != list2 and list1[len(list1) - 1] == list2[len(list2) - 1]:
                            density1 = calculate_density(list1, input_image, origin)
                            density2 = calculate_density(list2, input_image, origin)
                            if density1 < density2 and len(list1) <= 4:
                                self.remove_pairs(list1[0], list1[1])
                            elif density2 < density1 and len(list2) <= 4:
                                self.remove_pairs(list2[0], list2[1])
                            done = True

    def remove_pairs(self, location_back, location_front):
        """This is a path-walking method used to remove nodes from the graph
        It is a helper function used by other methods in this file. Each call
        to this method will remove a single node from the graph."""
        node1 = self.get_node(location_back)
        node1.edges.remove(location_front)
        node2 = self.get_node(location_front)
        node2.edges.remove(location_back)
        num_edges = node2.get_num_edges()
        if num_edges == 1:
            self.remove_pairs(location_front, node2.get_edges()[0])  # Should only be one left

    def remove_side_chains(self):
        """This method removes sides chains from the graph
        This method looks for nodes that have three or more edges. It then
        calculates if a single trace connects to another node with three or
        more edges. If this is the case then this trace is likely a side chain
        shortcut that should not exist. It is then removed by this method."""
        for node in self.nodes:
            if node.get_num_edges() >= 3:  # This is a trinary connection
                visited = list()
                visited.append(node.get_location())
                min1 = 111  # Smallest
                min1_edge = None
                min2 = 999
                for edge in node.get_edges():
                    value = self.walk_graph(edge, deepcopy(visited), 1)
                    if value < min2 and value < min1:
                        min2 = min1
                        min1 = value
                        min1_edge = edge
                    elif value < min2:
                        min2 = value
                if min1 <= 3 <= min2 - min1:  # remove from graph
                    self.remove_pairs(node.get_location(), min1_edge)

    def remove_empty_nodes(self):
        """This function removes empty nodes in the graph. (Garbage collection)"""
        for node in self.nodes:
            if node.get_num_edges() == 0:
                self.nodes.remove(node)

    def print_traces(self, sheet_image, helix_image, offset, pdb_file):
        """This method prints all traces in the graph to a single .PDB file"""
        writer = open(pdb_file, 'w')
        already_written = list()
        set_of_traces = list()
        for node in self.nodes:
            if node.get_num_edges() == 1 or node.get_num_edges() > 2:
                for edge in node.get_edges():
                    trace = list()
                    trace.append(node.get_location())
                    previous = node
                    try:
                        cur = self.get_node(edge)
                        while True:
                            trace.append(cur.get_location())
                            if cur.get_num_edges() != 2:
                                break
                            else:
                                temp_node = cur
                                if cur.get_edges()[0] == previous.get_location():
                                    cur = self.get_node(cur.get_edges()[1])
                                else:
                                    cur = self.get_node(cur.get_edges()[0])
                                previous = temp_node
                    except ValueError as e:
                        print(e)

                    representation = repr(trace[0]) + repr(trace[len(trace) - 1])
                    if representation not in already_written:
                        set_of_traces.append(trace)
                        already_written.append(repr(trace[len(trace) - 1]) + repr(trace[0]))

        helix_traces = list()
        helix_chains = list()
        sheet_traces = list()
        sheet_chains = list()
        cur_chain = 0
        counter = 0
        for trace in set_of_traces:
            cur_chain += 1
            cur_helix = None
            cur_sheet = None
            for ca in trace:
                ca_orig = [int(ca[2] - offset[2]), int(ca[1] - offset[1]), int(ca[0] - offset[0])]
                if helix_image[ca_orig[0]][ca_orig[1]][ca_orig[2]] > 0:
                    if cur_helix is None:
                        cur_helix = list()
                        cur_helix.append(counter)
                    else:
                        cur_helix.append(counter)
                else:
                    if cur_helix is not None:
                        helix_traces.append(cur_helix)
                        helix_chains.append(cur_chain)
                        cur_helix = None
                if sheet_image[ca_orig[0]][ca_orig[1]][ca_orig[2]] > 0:
                    if cur_sheet is None:
                        cur_sheet = list()
                        cur_sheet.append(counter)
                    else:
                        cur_sheet.append(counter)
                else:
                    if cur_sheet is not None:
                        sheet_traces.append(cur_sheet)
                        sheet_chains.append(cur_chain)
                        cur_sheet = None

                PDB_Reader_Writer.write_single_pdb(file=writer, type='ATOM', chain='A',
                                                   node=np.array([ca[0], ca[1], ca[2]]), seqnum=(counter + cur_chain))
                counter += 1
            PDB_Reader_Writer.write_single_pdb(file=writer, type='TER')

            if cur_helix is not None:  # Fence-Posting
                helix_traces.append(cur_helix)
                helix_chains.append(cur_chain)
            if cur_sheet is not None:  # Fence-Posting
                sheet_traces.append(cur_sheet)
                sheet_chains.append(cur_chain)

        for index, trace in enumerate(helix_traces):
            chain = helix_chains[index]
            start = str(helix_traces[index][0] + chain - 1)
            end = str(helix_traces[index][len(helix_traces[index]) - 1] + chain - 1)
            PDB_Reader_Writer.write_single_pdb(file=writer, type='HELIX', chain='A', node_from=start, node_to=end)

        for index, trace in enumerate(sheet_traces):
            chain = sheet_chains[index]
            start = str(sheet_traces[index][0] + chain - 1)
            end = str(sheet_traces[index][len(sheet_traces[index]) - 1] + chain - 1)
            PDB_Reader_Writer.write_single_pdb(file=writer, type='SHEET', chain='A', node_from=start, node_to=end)

        writer.close()

    def refine_backbone(self, backbone_image, origin):
        box_size = np.shape(backbone_image)
        new_backbone = np.zeros(box_size)
        steps = 10
        already_written = list()
        for node in self.nodes:
            for edge in node.get_edges():
                node_location = node.get_location()
                node_location = [int(node_location[2] - origin[2]),
                                 int(node_location[1] - origin[1]),
                                 int(node_location[0] - origin[0])]  # Reverse it
                edge_location = [int(edge[2] - origin[2]),
                                 int(edge[1] - origin[1]),
                                 int(edge[0] - origin[0])]  # Reverse it
                representation = repr(edge_location) + repr(node_location)
                if representation not in already_written:
                    already_written.append(repr(node_location) + repr(edge_location))
                    midpoints = list()
                    x_step = (node_location[0] - edge_location[0]) / steps
                    y_step = (node_location[1] - edge_location[1]) / steps
                    z_step = (node_location[2] - edge_location[2]) / steps
                    for index in range(steps + 1):
                        midpoints.append([edge_location[0] + x_step * index,
                                          edge_location[1] + y_step * index,
                                          edge_location[2] + z_step * index])
                    for z in range(int(edge_location[2]) - 4, int(edge_location[2]) + 4):  # 4 is kinda arbitrary here
                        for y in range(int(edge_location[1]) - 4, int(edge_location[1]) + 4):
                            for x in range(int(edge_location[0]) - 4, int(edge_location[0]) + 4):
                                placed = False
                                for index in range(len(midpoints)):
                                    if (box_size[2] > z >= 0 <= y < box_size[1] and 0 <= x < box_size[0] and
                                            distance(z, midpoints[index][2], y, midpoints[index][1], x,
                                                     midpoints[index][0]) <= 2
                                            and not placed):
                                        placed = True
                                        new_backbone[x][y][z] = backbone_image[x][y][z]
        new_backbone = np.array(new_backbone, dtype=np.float32)
        return new_backbone

    def print_graph(self, pdb_file):
        """This method prints the graph as a list of edges
        There is no ordering in the graph. It is merely a tool for getting a
        feel for each connection between each Ca atom."""
        writer = open(pdb_file, 'w')
        already_written = list()
        counter = 1
        for node in self.nodes:
            for edge in node.get_edges():
                node_location = node.get_location()
                representation = repr(edge) + repr(node_location)
                if representation not in already_written:
                    PDB_Reader_Writer.write_single_pdb(file=writer, type='ATOM', chain='A', node=np.array(
                        [node_location[0], node_location[1], node_location[2]]), seqnum=counter)
                    PDB_Reader_Writer.write_single_pdb(file=writer, type='ATOM', chain='A',
                                                       node=np.array([edge[0], edge[1], edge[2]]), seqnum=(counter + 1))
                    counter += 3
                    already_written.append(repr(node_location) + repr(edge))
        writer.close()


def calculate_density(walk_list, full_image, origin):
    """Calculates the density of a trace
    This is used to determine which trace should be removed when there are
    more than two traces coming out of a node"""
    box_size = np.shape(full_image)
    steps = 10
    total_density = 0
    number_of_points = 0
    for i in range(len(walk_list) - 1):
        start_point_trans = [walk_list[i][2] - origin[2],
                             walk_list[i][1] - origin[1],
                             walk_list[i][0] - origin[0]]
        end_point_trans = [walk_list[i + 1][2] - origin[2],
                           walk_list[i + 1][1] - origin[1],
                           walk_list[i + 1][0] - origin[0]]
        midpoints = list()
        x_step = (end_point_trans[0] - start_point_trans[0]) / steps
        y_step = (end_point_trans[1] - start_point_trans[1]) / steps
        z_step = (end_point_trans[2] - start_point_trans[2]) / steps
        for j in range(steps + 1):
            midpoints.append([start_point_trans[0] + x_step * j,
                              start_point_trans[1] + y_step * j,
                              start_point_trans[2] + z_step * j])
        for z in range(int(start_point_trans[2]) - 4, int(start_point_trans[2]) + 4):  # 4 is kinda arbitrary here
            for y in range(int(start_point_trans[1]) - 4, int(start_point_trans[1]) + 4):
                for x in range(int(start_point_trans[0]) - 4, int(start_point_trans[0]) + 4):
                    placed = False
                    for j in range(len(midpoints)):
                        if (box_size[2] > z >= 0 <= y < box_size[1] and 0 <= x < box_size[0] and
                                distance(z, midpoints[j][2], y, midpoints[j][1], x, midpoints[j][0]) <= 1
                                and not placed):
                            placed = True
                            total_density += full_image[x][y][z]
                            number_of_points += 1

    return total_density / number_of_points


def make_graph(pdb_file):
    """This function is not part of the Graph or Node class but it is
    essentially a wrapper method. This method is called to turn an input .PDB
    file into a Graph representation for later processing. This allows the
    prediction step to be separated from the post-processing step. They do not
    have to be run concurrently."""
    pdb_file = open(pdb_file, 'r')
    graph = Graph()
    previous_location = None
    cur_index = -1
    for line in pdb_file:
        if line.startswith("ATOM"):
            index = PDB_Reader_Writer.read_single_pdb_line(type='ATOM INDEX', line=line)
            x, y, z = PDB_Reader_Writer.read_single_pdb_line(type='ATOM', line=line)
            if index == cur_index + 1:
                previous = graph.get_node(previous_location)
                previous.add_edge([x, y, z])
                if graph.contains_location([x, y, z]):
                    node = graph.get_node([x, y, z])
                    node.add_edge(previous_location)
                else:
                    new_node = Node([x, y, z])
                    new_node.add_edge(previous.get_location())
                    graph.add_node(new_node)
            else:  # new chain
                if not graph.contains_location([x, y, z]):
                    new_node = Node([x, y, z])
                    graph.add_node(new_node)
            cur_index = index
            previous_location = [x, y, z]  # Update for next go-around
    return graph