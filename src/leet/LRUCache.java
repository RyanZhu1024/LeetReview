package leet;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by RyanZhu on 6/3/16.
 */
public class LRUCache {

    int capacity;
    Map<Integer,Node> map;
    Node tail;
    Node head;


    // @param capacity, an integer
    public LRUCache(int capacity) {
        // write your code here
        this.capacity = capacity;
        map = new HashMap<>();
        head = new Node(-1,-1);
        tail = new Node(-1,-1);
        head.next = tail;
        tail.prev = head;
    }

    // @return an integer
    public int get(int key) {
        // write your code here
        if(map.containsKey(key)){
            promote(map.get(key));
            return map.get(key).val;
        }else{
            return -1;
        }
    }

    // @param key, an integer
    // @param value, an integer
    // @return nothing
    public void set(int key, int value) {
        // write your code here
        if(map.containsKey(key)){
            map.get(key).val = value;
            promote(map.get(key));
        }else{
            if(map.size() == capacity){
                remove(tail.prev.key);
            }
            insert(key, value);
        }
    }

    private void promote(Node node){
        node.prev.next = node.next;
        node.next.prev = node.prev;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
        node.prev = head;
    }

    private void remove(int key){
        map.remove(key);
        tail.prev = tail.prev.prev;
        tail.prev.next = tail;
    }

    private void insert(int key, int value){
        Node node = new Node(key,value);
        node.next = head.next;
        node.prev = head;
        head.next.prev = node;
        head.next = node;
        map.put(key, node);
    }

    class Node{
        int val;
        Node prev;
        Node next;
        int key;

        public Node(int key,int val){
            this.key = key;
            this.val = val;
        }
    }

    public static void main(String[] args) {
        LRUCache c = new LRUCache(2);
        c.set(2,1);
        c.set(1,1);
        System.out.println(c.get(2));
        c.set(4,1);
        System.out.println(c.get(1));
        System.out.println(c.get(2));
    }
}
