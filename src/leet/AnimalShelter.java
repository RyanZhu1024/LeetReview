package leet;

/**
 * Created by rzhu on 6/21/16.
 */
public class AnimalShelter {

    Node head;
    Node tail;
    Node headdm;
    Node dogHead;
    Node dogdm;
    Node catHead;
    Node catdm;

    public AnimalShelter() {
        // do initialize if necessary
        head = new Node("dummy", -1);
        tail = new Node("tail",-1);
        head.next = tail;
        tail.prev = head;

        headdm = head;
        dogHead = new Node("dog",-1);
        dogdm = dogHead;
        catHead = new Node("cat",-1);
        catdm = catHead;
    }

    /**
     * @param name a string
     * @param type an integer, 1 if Animal is dog or 0
     * @return void
     */
    void enqueue(String name, int type) {
        // Write your code here
        Node node = new Node(name, type);
        node.dogNext = tail;
        node.catNext = tail;
        headdm.next = node;
        node.next = tail;
        tail.prev = node;
        node.prev = headdm;
        headdm = headdm.next;
        if(type == 1){
            dogdm.dogNext = node;
            dogdm = dogdm.dogNext;
        }else{
            catdm.catNext = node;
            catdm = catdm.catNext;
        }
    }

    public String dequeueAny() {
        // Write your code here
        String name;
        if(head.next.type == 1){
            name = dequeueDog();
        }else{
            name = dequeueCat();
        }
        return name;
    }

    public String dequeueDog() {
        // Write your code here
        Node node = dogHead.dogNext;
        String name = node.name;
        node.prev.next = node.next;
        node.next.prev = node.prev;
        dogHead.dogNext = node.dogNext;
        if(dogHead.dogNext == tail){
            dogdm = dogHead;
        }
        if(node.next == tail){
            headdm = node.prev;
        }
        System.out.println(name);
        return name;
    }

    public String dequeueCat() {
        // Write your code here
        Node node = catHead.catNext;
        String name = node.name;
        node.prev.next = node.next;
        node.next.prev = node.prev;
        catHead.catNext = node.catNext;
        if(catHead.catNext == tail){
            catdm = catHead;
        }
        if(node.next == tail){
            headdm = node.prev;
        }
        System.out.println(name);
        return name;
    }


    class Node{
        String name;
        int type;
        Node next;
        Node prev;
        Node dogNext;
        Node catNext;
        public Node(String n, int type){
            this.name = n;
            this.type = type;
        }
    }

    void print(AnimalShelter as){
        Node head = as.head;
        while(head.next != null){
            System.out.println(head.next.name);
            head = head.next;
        }
    }

    public static void main(String[] args) {
        AnimalShelter as = new AnimalShelter();
        as.enqueue("ajpy", 1);
        as.enqueue("wajb", 0);
        as.dequeueAny();
        as.enqueue("hjyw", 1);
        as.dequeueAny();
        as.enqueue("wtyw", 1);
        as.enqueue("jght", 1);
        as.enqueue("apwy", 0);
        as.dequeueCat();
        as.dequeueDog();
        as.enqueue("ybwg", 0);
        as.enqueue("jpwa", 1);
        as.dequeueCat();
        as.dequeueDog();
        as.enqueue("jayh", 1);
        as.enqueue("atww", 0);
        as.dequeueDog();
        as.enqueue("wjpt", 0);
        as.dequeueCat();
        as.dequeueDog();
        as.enqueue("yhwp", 0);
        as.enqueue("gwya", 1);
        as.dequeueCat();
        as.dequeueCat();
        as.enqueue("jgwb", 0);
        as.enqueue("agyp", 1);
        as.dequeueDog();
        as.dequeueCat();
        as.enqueue("pbtw", 1);
        as.dequeueDog();
        as.enqueue("wgjy", 0);
        as.enqueue("gbat", 0);
        as.dequeueAny();
        as.enqueue("ahbw", 0);
        as.dequeueDog();
        as.dequeueCat();
        as.dequeueCat();
        as.enqueue("wbya", 0);
        as.dequeueCat();
        as.enqueue("pgty", 0);
    }
}

