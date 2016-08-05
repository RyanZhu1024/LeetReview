package leet;

import java.util.Deque;
import java.util.LinkedList;

/**
 * Created by rzhu on 7/28/16.
 */
public class Google {
    public static void main(String[] args) {
        Google g = new Google();
        System.out.println(g.findMaxOA13(1234564));
    }
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return new int[0];
        }
        Deque<Integer> queue = new LinkedList<>();
        int[] maxes = new int[nums.length - k + 1];
        int idx = 0;
        for (int i = 0; i < nums.length; i++) {
            while (!queue.isEmpty() && queue.peek() < i - k + 1) {
                queue.poll();
            }
            while (!queue.isEmpty() && nums[queue.peekLast()] < nums[i]) {
                queue.pollLast();
            }
            queue.offer(i);
            if (i >= k - 1) {
                maxes[idx++] = nums[queue.peek()];
            }
        }
        return maxes;
    }

    public int findMaxOA11(int num) {
        // delete one digit from identical adjacent digits
        int max = 0;
        String numStr = String.valueOf(num);
        int i = 0;
        while (i < numStr.length() - 1) {
            int j = i;
            while (j < numStr.length() - 1 && numStr.charAt(j + 1) == numStr.charAt(i)) {
                j++;
            }
            if (j > i) {
                String p1 = numStr.substring(0, i);
                String p2 = numStr.substring(i, j);
                String p3 = numStr.substring(j + 1);
                max = Math.max(max, Integer.parseInt(p1 + p2 + p3));
            }
            i = j + 1;
        }
        return max;
    }

    public int findMaxOA12(int num) {
        // repeat one digit and get the max value
        if (num == 0) {
            return 0;
        } else {
            String numstr = String.valueOf(num);
            int max = 0;
            for (int i = 0; i < numstr.length(); i++) {
                if (i == numstr.length() - 1) {
                    max = Integer.parseInt(numstr + numstr.charAt(i));
                } else if (numstr.charAt(i) > numstr.charAt(i + 1)){
                    max = Integer.parseInt(numstr.substring(0, i + 1) +
                            numstr.charAt(i) + numstr.substring(i + 1));
                    break;
                }
            }
            return max;
        }
    }

    public int findMaxOA13(int num) {
        //delete adjacent greater digit and get max
        if (num == 0) {
            return 0;
        } else {
            String numstr = String.valueOf(num);
            char preChar = numstr.charAt(0);
            int max = 0;
            for (int i = 1; i < numstr.length(); i++) {
                char curChar = numstr.charAt(i);
                if (curChar > preChar) {
                    String temp = numstr.substring(0, i) + numstr.substring(i + 1);
                    max = Math.max(max, Integer.parseInt(temp));
                } else {
                    String temp = numstr.substring(0, i - 1) + numstr.substring(i);
                    max = Math.max(max, Integer.parseInt(temp));
                }
            }
            return max;
        }
    }
}
