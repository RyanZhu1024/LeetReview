package leet;

import java.util.Stack;

/**
 * Created by RyanZhu on 6/2/16.
 */
public class Stack2Queue {
    private Stack<Integer> stack1;
    private Stack<Integer> stack2;

    public Stack2Queue() {
        // do initialization if necessary
        stack1 = new Stack<>();
        stack2 = new Stack<>();
    }

    public void push(int element) {
        // write your code here
        stack1.push(element);
    }

    public int pop() {
        // write your code here
        move();
        return stack2.pop();
    }

    public int top() {
        // write your code here
        move();
        return stack2.peek();
    }

    void move(){
        while(!stack1.isEmpty()){
            stack2.push(stack1.pop());
        }
    }

    public static void main(String[] args) {
        Stack2Queue q = new Stack2Queue();
        q.push(1); q.push(2); q.push(3); q.push(4); q.push(5);
        System.out.println(q.pop());
        System.out.println(q.pop());
        q.push(6); q.push(7); q.push(8); q.push(9);
        System.out.println(q.pop());
        System.out.println(q.pop());
    }

}
