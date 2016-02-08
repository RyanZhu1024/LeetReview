package leet;

import java.util.Stack;

/**
 * Created by RyanZhu on 11/23/15.
 */
public class MinStack {
    Stack<Integer> stack = new Stack<>();
    int min = Integer.MAX_VALUE;

    public void push(int x) {
        if (x < min) {
            min = x;
        }
        stack.push(x);
    }

    public void pop() {
        int ele = stack.pop();
        if (ele == min) {
            min = Integer.MAX_VALUE;
            for (Integer integer : stack) {
                if (integer < min) {
                    min = integer;
                }
            }
        }
    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return min;
    }
}
