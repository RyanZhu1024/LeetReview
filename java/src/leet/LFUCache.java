package leet;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by rzhu on 6/21/16.
 */
public class LFUCache {

    Map<Integer,Node> map;
    int cap;
    int max = 1;

    Node head, tail, insertionPoint;

    // @param capacity, an integer
    public LFUCache(int capacity) {
        // Write your code here
        map = new HashMap<>(capacity);
        head = new Node(0, 0, 0);
        tail = new Node(0, 0, 0);
        head.next = tail;
        tail.prev = head;
        this.cap = capacity;
        insertionPoint = tail;
    }

    // @param key, an integer
    // @param value, an integer
    // @return nothing
    public void set(int key, int value) {
        // Write your code here
         if(map.containsKey(key)){
             map.get(key).freq += 1;
             map.get(key).value = value;
             promoteNode(map.get(key));
         }else{
        Node node = new Node(key,value, 1);
        map.put(key, node);
        if(map.size() > cap){
            Node toDel = tail.prev;
            if(toDel == insertionPoint){
                insertionPoint = toDel.next;
            }
            toDel.prev.next = toDel.next;
            toDel.next.prev = toDel.prev;
            toDel.next = null;
            toDel.prev = null;
            map.remove(toDel.key);
            if(toDel.freq == max){
                max = toDel.next.freq;
            }
        }
        insert(node);
         }
    }

    void insert(Node node){
        node.next = insertionPoint;
        node.prev = insertionPoint.prev;
        insertionPoint.prev.next = node;
        insertionPoint.prev = node;
        insertionPoint = node;
    }


    public int get(int key) {
        // Write your code here
        if(map.containsKey(key)){
            Node node = map.get(key);
            node.freq += 1;
            if(node == insertionPoint && node.freq > node.next.freq){
                insertionPoint = node.next;
            }
            promoteNode(node);
            max = Math.max(max, node.freq);
            System.out.print(node.value + ", ");
            return node.value;
        }else{
            System.out.print(-1 + ", ");
            return -1;
        }
    }

    void promoteNode(Node node){
        // if(node.prev != head && node.freq >= max){
        //     node.prev.next = node.next;
        //     node.next.prev = node.prev;
        //     node.prev = head;
        //     node.next = head.next;
        //     head.next.prev = node;
        //     head.next = node;
        // }
        while(node.prev != head && node.freq >= node.prev.freq){
            Node prevNode = node.prev;
            Node nextNode = node.next;
            node.next = prevNode;
            node.prev = prevNode.prev;
            prevNode.prev.next = node;
            prevNode.prev = node;
            prevNode.next = nextNode;
            nextNode.prev = prevNode;
        }
    }

    class Node{
        int value;
        int key;
        int freq;
        Node prev;
        Node next;
        public Node(int k, int v, int f){
            this.key = k;
            this.value = v;
            this.freq = f;
        }
    }


    public static void main(String[] args) {
        LFUCache lfuCache = new LFUCache(3);
        lfuCache.set(1,10);
        lfuCache.set(2,20);
        lfuCache.set(3,30);
        lfuCache.get(1);
        lfuCache.set(4,40);
        lfuCache.get(4);
        lfuCache.get(3);
        lfuCache.get(2);
        lfuCache.get(1);
        lfuCache.set(5,50);
        lfuCache.get(1);
        lfuCache.get(2);
        lfuCache.get(3);
        lfuCache.get(4);
        lfuCache.get(5);
    }
}
