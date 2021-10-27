package graph;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by RyanZhu on 11/12/15.
 */
public class Graph {
    List<Vertex> vertices=new ArrayList<>();
    List<Edge> edges=new ArrayList<>();
    public void resetVertices(){
        for (Vertex vertice : vertices) {
            vertice.visited=false;
        }
    }

    public void shortestPath(Vertex v1,Vertex v2){
        List<Vertex> result=new ArrayList<>();
        getParent(v1,v2,result);
        for (int i =result.size()-1; i >=0; --i) {
            System.out.println(result.get(i).value);
        }
    }

    private void getParent(Vertex v1,Vertex v2,List<Vertex> result){
        result.add(v2);
        if(!v2.equals(v1)){
            getParent(v1,v2.parent,result);
        }
    }

    public void printBfs(Vertex vertex){
        List<Vertex> frontiers=new ArrayList<>();
        if(!vertex.visited) {
            vertex.visited = true;
            System.out.print(vertex.value);
        }
        if(vertex.adccents!=null&&vertex.adccents.size()>0){
            for (Vertex adccent : vertex.adccents) {
                if(!adccent.visited) {
                    adccent.parent=vertex;
                    System.out.print(adccent.value);
                    adccent.visited=true;
                    frontiers.add(adccent);
                }
            }
            for (Vertex frontier : frontiers) {
                printBfs(frontier);
            }
        }
    }

    public void printDfs(){
        for (Vertex start : vertices) {
            if(!start.visited) {
                printDfsVisit(start);
            }
        }
    }

    private void printDfsVisit(Vertex vertex){
        if(!vertex.visited){
            System.out.print(vertex.value);
            vertex.visited=true;
            for (Vertex adccent : vertex.adccents) {
                printDfsVisit(adccent);
            }
        }else{
            return;
        }
    }

    public static void main(String[] args) {
        Graph graph=new Graph();
        for (int i = 0; i < 10; i++) {
            Vertex vertex=new Vertex(i);
            graph.vertices.add(vertex);
        }
        graph.vertices.get(0)
                .linkVertexUndirected(graph.vertices.get(1))
                .linkVertexUndirected(graph.vertices.get(3))
                .linkVertexUndirected(graph.vertices.get(5));
        graph.vertices.get(1)
                .linkVertexUndirected(graph.vertices.get(2))
                .linkVertexUndirected(graph.vertices.get(9))
                .linkVertexUndirected(graph.vertices.get(8));
        graph.vertices.get(4)
                .linkVertexUndirected(graph.vertices.get(7))
                .linkVertexUndirected(graph.vertices.get(6))
                .linkVertexUndirected(graph.vertices.get(5))
                .linkVertexUndirected(graph.vertices.get(9));
        System.out.println();
        System.out.println("-----------------------bfs: start---------------------");
        graph.printBfs(graph.vertices.get(2));
        System.out.println();
        System.out.println("-----------------------bfs: end---------------------");
        System.out.println();
        System.out.println("-----------------------shortest path: start---------------------");
        graph.shortestPath(graph.vertices.get(2), graph.vertices.get(5));
        graph.printDfs();
        System.out.println();
        System.out.println("-----------------------shortest: end---------------------");
        System.out.println("-----------------------dfs: start---------------------");
        graph.resetVertices();
        graph.printDfs();
        System.out.println();
        System.out.println("-----------------------bfs: end---------------------");
    }
}
