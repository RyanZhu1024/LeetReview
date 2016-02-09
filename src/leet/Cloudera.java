package leet;

import java.util.Arrays;
import java.util.Stack;

/**
 * Created by RyanZhu on 2/8/16.
 */
public class Cloudera {
    public int solution(int[] A, int[] B) {
        // write your code in Java SE 8
        Arrays.sort(A);
        Arrays.sort(B);
        int i = 0, j = 0, result = -1;
        while (i < A.length && j < B.length) {
            if (A[i] < B[j]) {
                i++;
            } else if (A[i] > B[j]) {
                j++;
            } else {
                result = A[i];
                break;
            }
        }
        return result;
    }

    public int solution(String S) {
        // write your code in Java SE 8
        S="13+62+7+10+13*+*+4095";
        Stack<Integer> stack = new Stack<>();
        int result = -1, max = (int) (Math.pow(2, 12) - 1);
        for (int i = 0; i < S.length(); i++) {
            char c = S.charAt(i);
            if (c >= '0' && c <= '9') {
                stack.push(Character.getNumericValue(c));
            } else {
                if (stack.size() >= 2) {
                    int i1 = stack.pop();
                    int i2 = stack.pop();
                    if (c == '*') {
                        int prod = i1 * i2;
                        if (prod > max) return -1;
                        stack.push(prod);
                    }
                    if (c == '+') {
                        int sum = i1 + i2;
                        if (sum > max) return -1;
                        stack.push(sum);
                    }
                } else {
                    return result;
                }
            }
        }
        if (stack.size() > 0) {
            result = stack.peek();
        }
        return result;
    }

    public static void main(String[] args) {
        System.out.println((char) 0);
    }
}
