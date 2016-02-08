package leet;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by RyanZhu on 11/22/15.
 * Implement the following operations of a stack using queues.
 * <p>
 * push(x) -- Push element x onto stack.
 * pop() -- Removes the element on top of the stack.
 * top() -- Get the top element.
 * empty() -- Return whether the stack is empty.
 * Notes:
 * You must use only standard operations of a queue -- which means only push to back, peek/pop from front, size, and is empty operations are valid.
 * Depending on your language, queue may not be supported natively. You may simulate a queue by using a list or deque (double-ended queue), as long as you use only standard operations of a queue.
 * You may assume that all operations are valid (for example, no pop or top operations will be called on an empty stack).
 * Update (2015-06-11):
 * The class name of the Java function had been updated to MyStack instead of Stack.
 */
public class MyStack {
    // Push element x onto stack.
    List<Integer> queue = new LinkedList<>();

    public void push(int x) {
        queue.add(0, x);
    }

    // Removes the element on top of the stack.
    public void pop() {
        queue.remove(0);
    }

    // Get the top element.
    public int top() {
        return queue.get(0);
    }

    // Return whether the stack is empty.
    public boolean empty() {
        return queue.isEmpty();
    }
}
