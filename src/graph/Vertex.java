package graph;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by RyanZhu on 11/12/15.
 */
public class Vertex {
    int value;
    boolean visited=false;
    Vertex parent;

    List<Vertex> adccents=new ArrayList<>();

    public Vertex(int value) {
        this.value = value;
    }

    public Vertex linkVertexUndirected(Vertex vertex){
        if(!adccents.contains(vertex)) {
            adccents.add(vertex);
            vertex.linkVertexUndirected(this);
        }
        return this;
    }
}
