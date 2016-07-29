package leet;

import java.util.Deque;
import java.util.LinkedList;

/**
 * Created by rzhu on 7/28/16.
 */
public class Google {
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
}
