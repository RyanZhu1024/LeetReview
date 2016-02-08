package leet;

/**
 * Created by RyanZhu on 1/20/16.
 */
public class Hard {
    /**
     * Suppose a sorted array is rotated at some pivot unknown to you beforehand.
     * <p>
     * (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
     * <p>
     * You are given a target value to search. If found in the array return its index, otherwise return -1.
     * <p>
     * You may assume no duplicate exists in the array.
     *
     * @param nums
     * @param target
     * @return
     */
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) return -1;
        if (nums.length == 1) return nums[0] == target ? 0 : -1;
        int head = 0, tail = nums.length - 1;
        while (head < tail) {
            int mid = (head + tail) >> 1;
            if (nums[mid] == target) return mid;
            if (nums[head] <= nums[mid]) {
                if (target >= nums[head] && target < nums[mid]) {
                    tail = mid - 1;
                } else {
                    head = mid + 1;
                }
            } else {
                if (target <= nums[tail] && target > nums[mid]) {
                    head = mid + 1;
                } else {
                    tail = mid - 1;
                }
            }
        }
        return nums[head] == target ? head : -1;
    }
}
