package leet;

import java.util.*;

/**
 * Created by RyanZhu on 12/13/15.
 */
public class Medium {
    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    // Definition for a binary tree node.
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    //     Definition for singly-linked list.
    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }

    }

    /**
     * Given an array of integers, every element appears twice except for one. Find that single one.
     * <p>
     * Note:
     * Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
     *
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        int result = 0;
        for (int num : nums) {
            result = result ^ num;
        }
        return result;
    }

    /**
     * Given an array of numbers nums, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the two elements that appear only once.
     * <p>
     * For example:
     * <p>
     * Given nums = [1, 2, 1, 3, 2, 5], return [3, 5].
     * <p>
     * Note:
     * The order of the result is not important. So in the above example, [5, 3] is also correct.
     * Your algorithm should run in linear runtime complexity. Could you implement it using only constant space complexity?
     *
     * @param nums
     * @return
     */
    public int[] singleNumber3(int[] nums) {
        int diff = 0;
        for (int num : nums) {
            diff ^= num;
        }
        diff = Integer.highestOneBit(diff);

        int[] result = new int[2];
        Arrays.fill(result, 0);
        for (int num : nums) {
            if ((diff & num) == 0) {
                result[0] ^= num;
            } else {
                result[1] ^= num;
            }
        }
        return result;
    }

    /**
     * Say you have an array for which the ith element is the price of a given stock on day i.
     * <p>
     * Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
     *
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        int total = 0;
        for (int i = 0; i < prices.length - 1; i++) {
            if (prices[i + 1] > prices[i]) total += prices[i + 1] - prices[i];
        }

        return total;
    }

    public int maxProfitOnece(int[] prices) {
        if (prices == null || prices.length <= 1) return 0;
        int buy = prices[0], sell = prices[1];
        int max = sell - buy;
        for (int i = 2; i < prices.length; i++) {
            buy = Math.min(buy, prices[i - 1]);
            max = Math.max(prices[i] - buy, max);
        }
        return max;
    }

    /**
     * Given an array of n integers where n > 1, nums, return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].
     * <p>
     * Solve it without division and in O(n).
     * <p>
     * For example, given [1,2,3,4], return [24,12,8,6].
     *
     * @param nums
     * @return
     */
    public int[] productExceptSelf(int[] nums) {
        int[] result = new int[nums.length];
        result[0] = 1;
        for (int i = 1; i < nums.length; i++) {
            result[i] = result[i - 1] * nums[i - 1];
        }
        int right = 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            result[i] *= right;
            right = right * nums[i];
        }
        return result;
    }

    /**
     * There are n bulbs that are initially off. You first turn on all the bulbs. Then, you turn off every second bulb. On the third round, you toggle every third bulb (turning on if it's off or turning off if it's on). For the nth round, you only toggle the last bulb. Find how many bulbs are on after n rounds.
     * <p>
     * Example:
     * <p>
     * Given n = 3.
     * <p>
     * At first, the three bulbs are [off, off, off].
     * After first round, the three bulbs are [on, on, on].
     * After second round, the three bulbs are [on, off, on].
     * After third round, the three bulbs are [on, off, off].
     * <p>
     * So you should return 1, because there is only one bulb is on.
     * We know all numbers factors are even number unless it's a square number.
     *
     * @param n
     * @return
     */
    public int bulbSwitch(int n) {
        return (int) Math.sqrt(n);
    }

    /**
     * Given a binary tree, return the preorder traversal of its nodes' values.
     * <p>
     * For example:
     * Given binary tree {1,#,2,3},
     * 1
     * \
     * 2
     * /
     * 3
     * return [1,2,3].
     * <p>
     * Note: Recursive solution is trivial, could you do it iteratively?
     *
     * @param root
     * @return
     */
    public List<Integer> preorderTraversal(Easy.TreeNode root) {
        List<Integer> result = new ArrayList<>();
        Stack<Easy.TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.empty()) {
            Easy.TreeNode treeNode = stack.pop();
            if (treeNode != null) {
                result.add(treeNode.val);
                stack.push(treeNode.right);
                stack.push(treeNode.left);
            }
        }
        return result;
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();

        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;

        while (cur != null || !stack.empty()) {
            while (cur != null) {
                stack.add(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            list.add(cur.val);
            cur = cur.right;
        }

        return list;
    }

    /**
     * Given a string array words, find the maximum value of length(word[i]) * length(word[j]) where the two words do not share common letters. You may assume that each word will contain only lower case letters. If no such two words exist, return 0.
     * <p>
     * Example 1:
     * Given ["abcw", "baz", "foo", "bar", "xtfn", "abcdef"]
     * Return 16
     * The two words can be "abcw", "xtfn".
     * <p>
     * Example 2:
     * Given ["a", "ab", "abc", "d", "cd", "bcd", "abcd"]
     * Return 4
     * The two words can be "ab", "cd".
     * <p>
     * Example 3:
     * Given ["a", "aa", "aaa", "aaaa"]
     * Return 0
     * No such pair of words.
     *
     * @param words
     * @return
     */
    public int maxProduct(String[] words) {
        int max = 0;
        int[] bytes = new int[words.length];
        for (int i = 0; i < words.length; i++) {
            int val = 0;
            for (int j = 0; j < words[i].length(); j++) {
                val |= 1 << (int) words[i].charAt(j) - 'a';//only take the 1<<count bit
            }
            bytes[i] = val;
        }
        for (int i = 0; i < bytes.length; i++) {
            for (int j = i + 1; j < bytes.length; j++) {
                if ((bytes[i] & bytes[j]) == 0) max = Math.max(max, words[i].length() * words[j].length());
            }
        }
        return max;
    }

    public boolean hasCycle(ListNode head) {
        if (head == null) return false;
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) return true;
        }
        return false;
    }

    /**
     * Given an integer, convert it to a roman numeral.
     * <p>
     * Input is guaranteed to be within the range from 1 to 3999.
     *
     * @param num
     * @return
     */
    public String intToRoman(int num) {
        int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] strs = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};

        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < values.length; i++) {
            while (num >= values[i]) {
                num -= values[i];
                sb.append(strs[i]);
            }
        }
        return sb.toString();
    }

    /**
     * Given an array of integers, every element appears three times except for one. Find that single one.
     * <p>
     * Note:
     * Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
     *
     * @param nums
     * @return
     */
    public int singleNumber2(int[] nums) {
        int a = 0;
        int b = 0;
        for (int c : nums) {
            int ta = (~a & b & c) | (a & ~b & ~c);
            b = (~a & ~b & c) | (~a & b & ~c);
            a = ta;
        }
        //we need find the number that is 01,10 => 1, 00 => 0.
        return a | b;
    }

    /**
     * Given n, how many structurally unique BST's (binary search trees) that store values 1...n?
     * <p>
     * For example,
     * Given n = 3, there are a total of 5 unique BST's.
     * <p>
     * 1         3     3      2      1
     * \       /     /      / \      \
     * 3     2     1      1   3      2
     * /     /       \                 \
     * 2     1         2                 3
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        int[] G = new int[n + 1];
        G[0] = G[1] = 1;

        for (int i = 2; i <= n; ++i) {
            for (int j = 1; j <= i; ++j) {
                G[i] += G[j - 1] * G[i - j];
            }
        }

        return G[n];
    }

    public class TreeLinkNode {
        int val;
        TreeLinkNode left, right, next;

        TreeLinkNode(int x) {
            val = x;
        }
    }

    public void connect(TreeLinkNode root) {
        if (root == null) return;
        TreeLinkNode cur = null;
        TreeLinkNode pre = root;
        while (pre.left != null) {
            cur = pre;
            while (cur != null) {
                cur.left.next = cur.right;
                if (cur.next != null) {
                    cur.right.next = cur.next.left;
                }
                cur = cur.next;
            }
            pre = pre.left;
        }
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums.length > 0) {
            TreeNode root = construct(0, nums.length - 1, nums);
            return root;
        } else {
            return null;
        }
    }

    public TreeNode construct(int low, int high, int... nums) {
        if (low > high) return null;
        int mid = (low + high) / 2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = construct(low, mid - 1, nums);
        node.right = construct(mid + 1, high, nums);
        return node;
    }

    /**
     * Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
     * <p>
     * For example, given the array [−2,1,−3,4,−1,2,1,−5,4],
     * the contiguous subarray [4,−1,2,1] has the largest sum = 6.
     * <p>
     * click to show more practice.
     * <p>
     * More practice:
     * If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.
     *
     * @param nums
     * @return
     */
    public int maxSubArray(int[] nums) {
        if (nums == null) return 0;
        if (nums.length == 1) return nums[0];
        int max = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] + nums[i - 1] > nums[i]) nums[i] += nums[i - 1];
            if (nums[i] > max) max = nums[i];
        }
        return max;
    }

    public int maxSubArrayLogn(int[] nums) {
        return divide(nums, 0, nums.length - 1, Integer.MIN_VALUE);
    }

    public int divide(int[] nums, int start, int end, int tmax) {
        if (start > end) {
            return Integer.MIN_VALUE;
        }
        int mid = start + (end - start) / 2;
        int lmax = divide(nums, start, mid - 1, tmax);
        int rmax = divide(nums, mid + 1, end, tmax);
        tmax = Math.max(tmax, Math.max(lmax, rmax));
        int sum = 0, mlmax = 0;
        for (int i = mid - 1; i >= start; i--) {
            sum += nums[i];
            mlmax = Math.max(mlmax, sum);
        }
        sum = 0;
        int mrmax = 0;
        for (int i = mid + 1; i <= end; i++) {
            sum += nums[i];
            mrmax = Math.max(mrmax, sum);
        }
        return Math.max(tmax, nums[mid] + mlmax + mrmax);
    }

    /**
     * Suppose a sorted array is rotated at some pivot unknown to you beforehand.
     * <p>
     * (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
     * <p>
     * Find the minimum element.
     * <p>
     * You may assume no duplicate exists in the array.
     *
     * @param nums
     * @return
     */
    public int findMin(int[] nums) {
        if (nums.length == 1) return nums[0];
        if (nums.length == 2) return Math.min(nums[0], nums[1]);
        int left = 0, right = nums.length - 1;
        while (left < right) {
            if (nums[left] < nums[right]) {
                return nums[left];
            }
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[left]) {
                right = mid;
            } else {
                left = mid;
            }
        }
        return Math.min(nums[left], nums[right]);
    }

    /**
     * Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
     * <p>
     * For example, given n = 3, a solution set is:
     * <p>
     * "((()))", "(()())", "(())()", "()(())", "()()()"
     *
     * @param n
     * @return
     */
    public List<String> generateParenthesis(int n) {
        List<String> pars = new ArrayList<>();
        addingPar(pars, "", n, 0);
        return pars;
    }

    void addingPar(List<String> pars, String str, int n, int m) {
        if (n == 0 && m == 0) {
            pars.add(str);
            return;
        }
        if (m > 0) addingPar(pars, str + ")", n, m - 1);
        if (n > 0) addingPar(pars, str + "(", n - 1, m + 1);
    }

    /**
     * Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.
     * <p>
     * Note:
     * You may assume k is always valid, 1 ≤ k ≤ BST's total elements.
     * <p>
     * Follow up:
     * What if the BST is modified (insert/delete operations) often and you need to find the kth smallest frequently? How would you optimize the kthSmallest routine?
     *
     * @param root
     * @param k
     * @return
     */
    public int kthSmallest(TreeNode root, int k) {
        return find(root, k).val;
    }

    int kthCount = 0;

    TreeNode find(TreeNode root, int k) {
        if (kthCount == k) return root;
        else {
            TreeNode node = null;
            if (root.left != null) node = find(root.left, k);
            if (kthCount == k) return node;
            kthCount++;
            if (kthCount == k) return root;
            else if (root.right != null) {
                return find(root.right, k);
            }
            return null;
        }
    }

    /**
     * The gray code is a binary numeral system where two successive values differ in only one bit.
     * <p>
     * Given a non-negative integer n representing the total number of bits in the code, print the sequence of gray code. A gray code sequence must begin with 0.
     * <p>
     * For example, given n = 2, return [0,1,3,2]. Its gray code sequence is:
     * <p>
     * 00 - 0
     * 01 - 1
     * 11 - 3
     * 10 - 2
     * Note:
     * For a given n, a gray code sequence is not uniquely defined.
     * <p>
     * For example, [0,2,3,1] is also a valid gray code sequence according to the above definition.
     * <p>
     * For now, the judge is able to judge based on one instance of gray code sequence. Sorry about that.
     */
    public List<Integer> grayCode(int n) {
        ArrayList<Integer> arr = new ArrayList<>();
        arr.add(0);
        for (int i = 0; i < n; i++) {
            int inc = 1 << i;
            for (int j = arr.size() - 1; j >= 0; j--) {
                arr.add(arr.get(j) + inc);
            }
        }
        return arr;
    }

    /**
     * A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
     * <p>
     * The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).
     * <p>
     * How many possible unique paths are there?
     *
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        int[][] table = new int[m][n];
        table[0][0] = 1;
        for (int i = 1; i < m; i++) {
            table[i][0] = 1;
        }
        for (int i = 1; i < n; i++) {
            table[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                table[i][j] = table[i - 1][j] + table[i][j - 1];
            }
        }
        return table[m - 1][n - 1];
    }

    /**
     * Say you have an array for which the ith element is the price of a given stock on day i.
     * <p>
     * Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times) with the following restrictions:
     * <p>
     * You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
     * After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)
     * Example:
     * <p>
     * prices = [1, 2, 3, 0, 2]
     * maxProfit = 3
     * transactions = [buy, sell, cooldown, buy, sell]
     * buy[i] = max(sell[i-2]-price, buy[i-1])
     * sell[i] = max(buy[i-1]+price, sell[i-1])
     *
     * @param prices
     * @return
     */
    public int maxProfitWithCooldown(int[] prices) {
        int sell = 0, prev_sell = 0, buy = Integer.MIN_VALUE, prev_buy;
        for (int price : prices) {
            prev_buy = buy;
            buy = Math.max(prev_sell - price, prev_buy);
            prev_sell = sell;
            sell = Math.max(prev_buy + price, prev_sell);
        }
        return sell;
    }

    /**
     * Given a collection of distinct numbers, return all possible permutations.
     * <p>
     * For example,
     * [1,2,3] have the following permutations:
     * [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], and [3,2,1].
     * to permute n, permute n-1 first
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new LinkedList<>();
        if (nums == null || nums.length == 0) return result;
        List<Integer> cur = new LinkedList<>();
        boolean visited[] = new boolean[nums.length];
        Arrays.fill(visited, false);
        permuteDFS(result, cur, nums, visited);
        return result;
    }

    private void permuteDFS(List<List<Integer>> result, List<Integer> cur, int[] nums, boolean[] visited) {
        if (cur.size() == nums.length) {
            result.add(new LinkedList<>(cur));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (!visited[i]) {
                ((LinkedList) cur).offer(nums[i]);
                visited[i] = true;
                permuteDFS(result, cur, nums, visited);
                visited[i] = false;
                ((LinkedList) cur).removeLast();
            }
        }
    }


    /**
     * Given a linked list, swap every two adjacent nodes and return its head.
     * <p>
     * For example,
     * Given 1->2->3->4, you should return the list as 2->1->4->3.
     * <p>
     * Your algorithm should use only constant space. You may not modify the values in the list, only nodes itself can be changed.
     *
     * @param head
     * @return
     */
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode n1 = head;
        ListNode n2 = head.next;
        int temp = n1.val;
        n1.val = n2.val;
        n2.val = temp;
        if (n2.next == null) return head;
        while (n1.next.next != null && n2.next.next != null) {
            n1 = n1.next.next;
            n2 = n2.next.next;
            int t1 = n1.val;
            n1.val = n2.val;
            n2.val = t1;
        }
        return head;
    }

    /**
     * Given an array with n objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.
     * <p>
     * Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.
     * <p>
     * Note:
     * You are not suppose to use the library's sort function for this problem.
     *
     * @param nums
     */
    public void sortColors(int[] nums) {
        int zeros = 0, ones = 0, twos = 0;
        for (int i = 0; i < nums.length; i++) {
            switch (nums[i]) {
                case 0:
                    zeros++;
                    break;
                case 1:
                    ones++;
                    break;
                case 2:
                    twos++;
                    break;
            }
        }
        int index = 0;
        for (int i = 0; i < zeros; i++) {
            nums[index++] = 0;
        }
        for (int i = 0; i < ones; i++) {
            nums[index++] = 1;
        }
        for (int i = 0; i < twos; i++) {
            nums[index++] = 2;
        }
    }

    public void sortColorsOnePass(int[] nums) {
        int zero = 0, two = nums.length - 1;
        for (int i = 0; i <= two; ) {
            if (nums[i] == 0) swap(nums, i++, zero++);
            else if (nums[i] == 2) swap(nums, two--, i);
            else i++;
        }
    }

    /**
     * Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.
     * <p>
     * Note: You can only move either down or right at any point in time.
     *
     * @param grid
     * @return
     */
    public int minPathSum(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] table = new int[m][n];
        table[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            table[i][0] = table[i - 1][0] + grid[i][0];
        }
        for (int i = 1; i < n; i++) {
            table[0][i] = table[0][i - 1] + grid[0][i];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                table[i][j] = Math.min(table[i - 1][j], table[i][j - 1]) + grid[i][j];
            }
        }
        return table[m - 1][n - 1];
    }

    /**
     * Given an integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.
     * <p>
     * For example,
     * Given n = 3,
     * <p>
     * You should return the following matrix:
     * [
     * [ 1, 2, 3 ],
     * [ 8, 9, 4 ],
     * [ 7, 6, 5 ]
     * ]
     *
     * @param n
     */
    public int[][] generateMatrix(int n) {
        int[][] ret = new int[n][n];
        int left = 0, top = 0;
        int right = n - 1, down = n - 1;
        int count = 1;
        while (left <= right) {
            for (int j = left; j <= right; j++) {
                ret[top][j] = count++;
            }
            top++;
            for (int i = top; i <= down; i++) {
                ret[i][right] = count++;
            }
            right--;
            for (int j = right; j >= left; j--) {
                ret[down][j] = count++;
            }
            down--;
            for (int i = down; i >= top; i--) {
                ret[i][left] = count++;
            }
            left++;
        }
        return ret;
    }

    /**
     * You are given an n x n 2D matrix representing an image.
     * <p>
     * Rotate the image by 90 degrees (clockwise).
     * <p>
     * Follow up:
     * Could you do this in-place?
     *
     * @param matrix
     */
    public void rotate(int[][] matrix) {
        int n = matrix.length - 1;
        int bound = matrix.length % 2 == 0 ? matrix.length / 2 - 1 : matrix.length / 2;
        int j = 0;
        while (j <= bound) {
            int i = j;
            while (i < n - j) {
                swap(matrix, j, i, i, n - j, n - i, j, n - j, n - i);
                i++;
            }
            j++;
        }
    }

    private void swap(int[][] matrix, int i, int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
        int val1 = matrix[i][i1];
        int val2 = matrix[i2][i3];
        int val3 = matrix[i4][i5];
        int val4 = matrix[i6][i7];
        matrix[i][i1] = val3;
        matrix[i2][i3] = val1;
        matrix[i4][i5] = val4;
        matrix[i6][i7] = val2;
    }

    /**
     * Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.
     * <p>
     * Note: You may not slant the container.
     *
     * @param height
     * @return
     */
    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1, maxArea = 0;
        while (left < right) {
            maxArea = Math.max(maxArea, Math.min(height[left], height[right]) * (right - left));
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return maxArea;
    }

    /**
     * Find all possible combinations of k numbers that add up to a number n, given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.
     * <p>
     * Ensure that numbers within the set are sorted in ascending order.
     * <p>
     * <p>
     * Example 1:
     * <p>
     * Input: k = 3, n = 7
     * <p>
     * Output:
     * <p>
     * [[1,2,4]]
     * <p>
     * Example 2:
     * <p>
     * Input: k = 3, n = 9
     * <p>
     * Output:
     * <p>
     * [[1,2,6], [1,3,5], [2,3,4]]
     * Credits:
     * Special thanks to @mithmatt for adding this problem and creating all test cases.
     *
     * @param k
     * @param n
     * @return
     */
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> result = new ArrayList<>();
        combinationSum3Helper(k, n, 1, new ArrayList<>(), result);
        return result;
    }

    public void combinationSum3Helper(int k, int n, int cur, List<Integer> list, List<List<Integer>> result) {
        if (k == 1) {
            if (n < cur || n > 9) return;
            list.add(n);
            result.add(list);
            return;
        }
        for (int i = cur; i <= n / k && i < 10; i++) {// n/k represents the starting number to the target n
            List<Integer> sub = new ArrayList<>(list);
            sub.add(i);
            combinationSum3Helper(k - 1, n - i, i + 1, sub, result);
        }
    }

    /**
     * According to the Wikipedia's article: "The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."
     * <p>
     * Given a board with m by n cells, each cell has an initial state live (1) or dead (0). Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):
     * <p>
     * Any live cell with fewer than two live neighbors dies, as if caused by under-population.
     * Any live cell with two or three live neighbors lives on to the next generation.
     * Any live cell with more than three live neighbors dies, as if by over-population..
     * Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
     * Write a function to compute the next state (after one update) of the board given its current state.
     * <p>
     * Follow up:
     * Could you solve it in-place? Remember that the board needs to be updated at the same time: You cannot update some cells first and then use their updated values to update other cells.
     * In this question, we represent the board using a 2D array. In principle, the board is infinite, which would cause problems when the active area encroaches the border of the array. How would you address these problems?
     *
     * @param board
     */
    public void gameOfLife(int[][] board) {
        if (board == null || board.length == 0) return;
        int m = board.length, n = board[0].length;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int lives = liveNeighbors(board, m, n, i, j);

                // In the beginning, every 2nd bit is 0;
                // So we only need to care about when the 2nd bit will become 1.
                if (board[i][j] == 1 && lives >= 2 && lives <= 3) {
                    board[i][j] = 3; // Make the 2nd bit 1: 01 ---> 11
                }
                if (board[i][j] == 0 && lives == 3) {
                    board[i][j] = 2; // Make the 2nd bit 1: 00 ---> 10
                }
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                board[i][j] >>= 1;  // Get the 2nd state.
            }
        }
    }

    public int liveNeighbors(int[][] board, int m, int n, int i, int j) {
        int topLeft = (i - 1 >= 0 && j - 1 >= 0) ? board[i - 1][j - 1] & 1 : 0;
        int top = (i - 1 >= 0 && j >= 0) ? board[i - 1][j] & 1 : 0;
        int topRight = (i - 1 >= 0 && j + 1 < n) ? board[i - 1][j + 1] & 1 : 0;
        int left = (i >= 0 && j - 1 >= 0) ? board[i][j - 1] & 1 : 0;
        int right = (i >= 0 && j + 1 < n) ? board[i][j + 1] & 1 : 0;
        int botLeft = (i + 1 < m && j - 1 >= 0) ? board[i + 1][j - 1] & 1 : 0;
        int bot = (i + 1 < m && j >= 0) ? board[i + 1][j] & 1 : 0;
        int botRight = (i + 1 < m && j + 1 < n) ? board[i + 1][j + 1] & 1 : 0;
        return top + topLeft + topRight + left + right + bot + botLeft + botRight;
    }

    /**
     * Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
     * <p>
     * Integers in each row are sorted from left to right.
     * The first integer of each row is greater than the last integer of the previous row.
     * For example,
     * <p>
     * Consider the following matrix:
     * <p>
     * [
     * [1,   3,  5,  7],
     * [10, 11, 16, 20],
     * [23, 30, 34, 50]
     * ]
     * Given target = 3, return true.
     *
     * @param matrix
     * @param target
     * @return
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int i = 0, j = matrix[0].length - 1;
        while (i < matrix.length && j >= 0) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] > target) {
                j--;
            } else {
                i++;
            }
        }

        return false;
    }


    public boolean searchMatrixBinary(int[][] matrix, int target) {
        if (matrix.length == 0) return false;
        int m = matrix.length, n = matrix[0].length;
        if (matrix[0][0] > target || matrix[m - 1][n - 1] < target) return false;
        int head = 0, tail = m - 1, mid = 0;
        while (head != tail && matrix[tail][0] > target) {
            mid = (head + tail + 1) / 2;
            if (matrix[mid][0] < target) head = mid;
            else if (matrix[mid][0] > target) tail = mid - 1;
            else return true;
        }
        int row = tail;
        head = 0;
        tail = matrix[0].length - 1;
        while (head <= tail) {
            mid = (head + tail) / 2;
            if (matrix[row][mid] < target) head = mid + 1;
            else if (matrix[row][mid] > target) tail = mid - 1;
            else return true;
        }
        return false;
    }

    /**
     * Given an unsorted array of integers, find the length of longest increasing subsequence.
     * <p>
     * For example,
     * Given [10, 9, 2, 5, 3, 7, 101, 18],
     * The longest increasing subsequence is [2, 3, 7, 101], therefore the length is 4. Note that there may be more than one LIS combination, it is only necessary for you to return the length.
     * <p>
     * Your algorithm should run in O(n2) complexity.
     * <p>
     * Follow up: Could you improve it to O(n log n) time complexity?
     *
     * @param nums
     * @return
     */
    public int lengthOfLIS(int[] nums) {
        int N = nums.length;
        if (N == 0) return 0;
        int[] dp = new int[N];
        Arrays.fill(dp, 1);
        int res = 1;
        for (int i = 1; i < N; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[j] + 1, dp[i]);
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    public int lengthOfLISLGN(int[] nums) {
        int[] dp = new int[nums.length];
        int len = 0;

        for (int x : nums) {
            int i = Arrays.binarySearch(dp, 0, len, x);
            if (i < 0) i = -(i + 1);
            dp[i] = x;
            if (i == len) len++;
        }

        return len;
    }

    /**
     * A peak element is an element that is greater than its neighbors.
     * <p>
     * Given an input array where num[i] ≠ num[i+1], find a peak element and return its index.
     * <p>
     * The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.
     * <p>
     * You may imagine that num[-1] = num[n] = -∞.
     * <p>
     * For example, in array [1, 2, 3, 1], 3 is a peak element and your function should return the index number 2.
     * <p>
     * click to show spoilers.
     * <p>
     * Note:
     * Your solution should be in logarithmic complexity.
     *
     * @param nums
     * @return
     */
    public int findPeakElement(int[] nums) {
        int N = nums.length;
        if (N == 1) {
            return 0;
        }

        int left = 0, right = N - 1;
        while (right - left > 1) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return (left == N - 1 || nums[left] > nums[left + 1]) ? left : right;
    }

    /**
     * Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place.
     * <p>
     * click to show follow up.
     * <p>
     * Follow up:
     * Did you use extra space?
     * A straight forward solution using O(mn) space is probably a bad idea.
     * A simple improvement uses O(m + n) space, but still not the best solution.
     * Could you devise a constant space solution?
     *
     * @param matrix
     */
    public void setZeroes(int[][] matrix) {
        int col0 = 1;
        for (int i = 0; i < matrix.length; i++) {
            if (matrix[i][0] == 0 && col0 == 1) col0 = 0;
            for (int j = 1; j < matrix[0].length; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        for (int i = matrix.length - 1; i >= 0; i--) {
            for (int j = matrix[0].length - 1; j >= 1; j--) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
            if (col0 == 0) matrix[i][0] = 0;
        }
    }

    public class BSTIterator {

        Stack<TreeNode> stack = new Stack<>();

        public BSTIterator(TreeNode root) {
            initStack(root);
        }

        public void initStack(TreeNode root) {
            if (root != null) {
                stack.push(root);
                initStack(root.left);
            }
        }

        /**
         * @return whether we have a next smallest number
         */
        public boolean hasNext() {
            return !stack.empty();
        }

        /**
         * @return the next smallest number
         */
        public int next() {
            TreeNode node = stack.pop();
            if (node.right != null) {
                initStack(node.right);
            }
            return node.val;
        }
    }

    class PeekingIterator implements Iterator<Integer> {

        Iterator<Integer> iter;
        int val;
        boolean hasNext = false;

        public PeekingIterator(Iterator<Integer> iterator) {
            // initialize any member here.
            if (iterator != null) {
                this.iter = iterator;
                this.hasNext = iter.hasNext();
                this.val = iterator.next();
            }
        }

        // Returns the next element in the iteration without advancing the iterator.
        public Integer peek() {
            return this.val;
        }

        // hasNext() and next() should behave the same as in the Iterator interface.
        // Override them if needed.
        @Override
        public Integer next() {
            int temp = val;
            if (this.iter.hasNext()) {
                this.val = iter.next();
            } else {
                this.hasNext = false;
            }
            return temp;
        }

        @Override
        public boolean hasNext() {
            return this.hasNext;
        }
    }

    /**
     * Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
     * <p>
     * Integers in each row are sorted in ascending from left to right.
     * Integers in each column are sorted in ascending from top to bottom.
     * For example,
     * <p>
     * Consider the following matrix:
     * <p>
     * [
     * [1,   4,  7, 11, 15],
     * [2,   5,  8, 12, 19],
     * [3,   6,  9, 16, 22],
     * [10, 13, 14, 17, 24],
     * [18, 21, 23, 26, 30]
     * ]
     * Given target = 5, return true.
     * <p>
     * Given target = 20, return false.
     *
     * @param matrix
     * @param target
     * @return
     */
    public boolean searchMatrix2(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        if (matrix[0][0] > target || matrix[m - 1][n - 1] < target) return false;
        int row = 0, col = n - 1;
        while (row < m && col >= 0) {
            if (matrix[row][col] == target) return true;
            else if (matrix[row][col] > target) col--;
            else row++;
        }
        return false;
    }

    public boolean searchMatrix2Log(int[][] matrix, int target) {
        int m = matrix.length;
        for (int i = 0; i < m; i++) {
            if (binarySearchHelper(matrix[i], target)) return true;
        }
        return false;
    }

    public boolean binarySearchHelper(int[] arr, int target) {
        int head = 0, tail = arr.length - 1, mid;
        while (head <= tail) {
            mid = head + (tail - head) / 2;
            if (arr[mid] == target) return true;
            else if (arr[mid] > target) tail = mid - 1;
            else head = mid + 1;
        }
        return false;
    }

    /**
     * Follow up for "Remove Duplicates":
     * What if duplicates are allowed at most twice?
     * <p>
     * For example,
     * Given sorted array nums = [1,1,1,2,2,3],
     * <p>
     * Your function should return length = 5, with the first five elements of nums being 1, 1, 2, 2 and 3. It doesn't matter what you leave beyond the new length.
     *
     * @param nums
     * @return
     */
    public int removeDuplicates(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int j = 0, counter = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == nums[j]) {
                counter++;
                if (counter < 2) {
                    nums[j + 1] = nums[i];
                    j++;
                } else {
                    continue;
                }
            } else {
                nums[j + 1] = nums[i];
                j++;
                counter = 0;
            }
        }
        return j + 1;
    }

    /**
     * Given an array of citations (each citation is a non-negative integer) of a researcher, write a function to compute the researcher's h-index.
     * <p>
     * According to the definition of h-index on Wikipedia: "A scientist has index h if h of his/her N papers have at least h citations each, and the other N − h papers have no more than h citations each."
     * <p>
     * For example, given citations = [3, 0, 6, 1, 5], which means the researcher has 5 papers in total and each of them had received 3, 0, 6, 1, 5 citations respectively. Since the researcher has 3 papers with at least 3 citations each and the remaining two with no more than 3 citations each, his h-index is 3.
     * <p>
     * Note: If there are several possible values for h, the maximum one is taken as the h-index.
     * <p>
     * Hint:
     * <p>
     * An easy approach is to sort the array first.
     * What are the possible values of h-index?
     * A faster approach is to use extra space.
     *
     * @param citations
     * @return
     */
    public int hIndex(int[] citations) {
        if (citations == null || citations.length == 0) return 0;
        Arrays.sort(citations);
        int i = 0, n = citations.length;
        while (i < n && citations[i] < n - i) i++;
        return n - i;
    }

    public int hIndexN(int[] citations) {
        int len = citations.length;
        int[] count = new int[len + 1];

        for (int c : citations)
            if (c > len)
                count[len]++;
            else
                count[c]++;


        int total = 0;
        for (int i = len; i >= 0; i--) {
            total += count[i];
            if (total >= i)
                return i;
        }

        return 0;
    }

    public int hIndex2(int[] citations) {
        int left = 0, len = citations.length, right = len - 1, mid;
        while (left <= right) {
            mid = (left + right) >> 1;
            if (citations[mid] == (len - mid)) return citations[mid];
            else if (citations[mid] > (len - mid)) right = mid - 1;
            else left = mid + 1;
        }
        return len - (right + 1);
    }

    /**
     * Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.
     * <p>
     * For example:
     * Given the following binary tree,
     * 1            <---
     * /   \
     * 2     3         <---
     * \     \
     * 5     4       <---
     * You should return [1, 3, 4].
     *
     * @param root
     * @return
     */
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        int counter = 1;
        rightSideView(root, result, counter);
        return result;
    }

    private void rightSideView(TreeNode root, List<Integer> result, int counter) {
        if (root != null) {
            if (counter > result.size()) {
                result.add(root.val);
            }
            counter++;
            rightSideView(root.right, result, counter);
            rightSideView(root.left, result, counter);
        }
    }

    /**
     * Given a string of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. The valid operators are +, - and *.
     * <p>
     * <p>
     * Example 1
     * Input: "2-1-1".
     * <p>
     * ((2-1)-1) = 0
     * (2-(1-1)) = 2
     * Output: [0, 2]
     * <p>
     * <p>
     * Example 2
     * Input: "2*3-4*5"
     * <p>
     * (2*(3-(4*5))) = -34
     * ((2*3)-(4*5)) = -14
     * ((2*(3-4))*5) = -10
     * (2*((3-4)*5)) = -10
     * (((2*3)-4)*5) = 10
     * Output: [-34, -14, -10, -10, 10]
     *
     * @param input
     * @return
     */
    public List<Integer> diffWaysToCompute(String input) {
        List<Integer> result = new ArrayList<>();
        if (input == null || input.length() == 0) return result;
        List<String> ops = new ArrayList<>();
        for (int i = 0; i < input.length(); i++) {
            int j = i;
            while (j < input.length() && Character.isDigit(input.charAt(j)))
                j++;
            String num = input.substring(i, j);
            ops.add(num);
            if (j != input.length()) ops.add(input.substring(j, j + 1));
            i = j;
        }
        result = compute(ops, 0, ops.size() - 1);
        return result;
    }

    private List<Integer> compute(List<String> ops, int lo, int hi) {
        List<Integer> result = new ArrayList<>();
        if (lo == hi) {
            Integer num = Integer.valueOf(ops.get(lo));
            result.add(num);
            return result;
        }
        for (int i = lo + 1; i <= hi - 1; i = i + 2) {
            String operator = ops.get(i);
            List<Integer> left = compute(ops, lo, i - 1), right = compute(ops, i + 1, hi);
            for (int leftNum : left)
                for (int rightNum : right) {
                    if (operator.equals("+"))
                        result.add(leftNum + rightNum);
                    else if (operator.equals("-"))
                        result.add(leftNum - rightNum);
                    else
                        result.add(leftNum * rightNum);
                }
        }
        return result;
    }

    public List<Integer> diffWaysToComputeDP(String input) {
        List<Integer> result = new ArrayList<>();
        if (input == null || input.length() == 0) return result;
        List<String> ops = new ArrayList<>();
        for (int i = 0; i < input.length(); i++) {
            int j = i;
            while (j < input.length() && Character.isDigit(input.charAt(j)))
                j++;
            String num = input.substring(i, j);
            ops.add(num);
            if (j != input.length()) ops.add(input.substring(j, j + 1));
            i = j;
        }
        int N = (ops.size() + 1) / 2; //num of integers
        ArrayList<Integer>[][] dp = (ArrayList<Integer>[][]) new ArrayList[N][N];
        for (int d = 0; d < N; d++) {
            if (d == 0) {
                for (int i = 0; i < N; i++) {
                    dp[i][i] = new ArrayList<>();
                    dp[i][i].add(Integer.valueOf(ops.get(i * 2)));
                }
                continue;
            }
            for (int i = 0; i < N - d; i++) {
                dp[i][i + d] = new ArrayList<>();
                for (int j = i; j < i + d; j++) {
                    ArrayList<Integer> left = dp[i][j], right = dp[j + 1][i + d];
                    String operator = ops.get(j * 2 + 1);
                    for (int leftNum : left)
                        for (int rightNum : right) {
                            if (operator.equals("+"))
                                dp[i][i + d].add(leftNum + rightNum);
                            else if (operator.equals("-"))
                                dp[i][i + d].add(leftNum - rightNum);
                            else
                                dp[i][i + d].add(leftNum * rightNum);
                        }
                }
            }
        }
        return dp[0][N - 1];
    }

    /**
     * Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.
     * <p>
     * An example is the root-to-leaf path 1->2->3 which represents the number 123.
     * <p>
     * Find the total sum of all root-to-leaf numbers.
     * <p>
     * For example,
     * <p>
     * 1
     * / \
     * 2   3
     * The root-to-leaf path 1->2 represents the number 12.
     * The root-to-leaf path 1->3 represents the number 13.
     * <p>
     * Return the sum = 12 + 13 = 25.
     *
     * @param root
     * @return
     */
    public int sumNumbers(TreeNode root) {
        List<Integer> nums = new ArrayList<>();
        traverse(root, nums, 0);
        int sum = 0;
        for (Integer num : nums) {
            sum += num;
        }
        return sum;
    }

    public void traverse(TreeNode root, List<Integer> list, int number) {
        if (root != null) {
            number = root.val + number * 10;
            if (root.left == null && root.right == null) {
                list.add(number);
            } else {
                traverse(root.left, list, number);
                traverse(root.right, list, number);
            }
        }
    }

    /**
     * Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
     * <p>
     * Note: Do not modify the linked list.
     * <p>
     * Follow up:
     * Can you solve it without using extra space?
     *
     * @param head
     * @return
     */
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;

            if (fast == slow) {
                ListNode slow2 = head;
                while (slow2 != slow) {
                    slow = slow.next;
                    slow2 = slow2.next;
                }
                return slow;
            }
        }
        return null;
    }

    /**
     * Follow up for "Search in Rotated Sorted Array": HARD
     * What if duplicates are allowed?
     * <p>
     * Would this affect the run-time complexity? How and why?
     * <p>
     * Write a function to determine if a given target is in the array.
     *
     * @param nums
     * @param target
     * @return
     */
    public boolean search(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (nums[m] == target) return true; //return m in Search in Rotated Array I
            if (nums[l] < nums[m]) { //left half is sorted
                if (nums[l] <= target && target < nums[m])
                    r = m - 1;
                else
                    l = m + 1;
            } else if (nums[l] > nums[m]) { //right half is sorted
                if (nums[m] < target && target <= nums[r])
                    l = m + 1;
                else
                    r = m - 1;
            } else l++;
        }
        return false;
    }

    /**
     * Given an integer matrix, find the length of the longest increasing path.
     * <p>
     * From each cell, you can either move to four directions: left, right, up or down. You may NOT move diagonally or move outside of the boundary (i.e. wrap-around is not allowed).
     * <p>
     * Example 1:
     * <p>
     * nums = [
     * [9,9,4],
     * [6,6,8],
     * [2,1,1]
     * ]
     * Return 4
     * The longest increasing path is [1, 2, 6, 9].
     * <p>
     * Example 2:
     * <p>
     * nums = [
     * [3,4,5],
     * [3,2,6],
     * [2,2,1]
     * ]
     * Return 4
     * The longest increasing path is [3, 4, 5, 6]. Moving diagonally is not allowed.
     *
     * @param matrix
     * @return
     */
    int max = 1;

    public int longestIncreasingPath(int[][] matrix) {
        if (matrix == null || matrix.length == 0) return 0;
        int m = matrix.length, n = matrix[0].length;
        int[][] record = new int[m][n];
        for (int i = 0; i < record.length; i++) {
            for (int j = 0; j < record[i].length; j++) {
                record[i][j] = 1;
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (record[i][j] == 1) {
                    depthSearch(i, j, matrix, record);
                }
            }
        }
        return max;
    }

    private void depthSearch(int i, int j, int[][] matrix, int[][] record) {
        int m = matrix.length, n = matrix[0].length;
        if (j > 0 && record[i][j - 1] <= record[i][j] && matrix[i][j - 1] > matrix[i][j]) {
            record[i][j - 1] = record[i][j] + 1;
            max = Math.max(max, record[i][j - 1]);
            depthSearch(i, j - 1, matrix, record);
        }
        if (j < n - 1 && record[i][j + 1] <= record[i][j] && matrix[i][j + 1] > matrix[i][j]) {
            record[i][j + 1] = record[i][j] + 1;
            max = Math.max(max, record[i][j + 1]);
            depthSearch(i, j + 1, matrix, record);
        }
        if (i < m - 1 && record[i + 1][j] <= record[i][j] && matrix[i + 1][j] > matrix[i][j]) {
            record[i + 1][j] = record[i][j] + 1;
            max = Math.max(max, record[i + 1][j]);
            depthSearch(i + 1, j, matrix, record);
        }
        if (i > 0 && record[i - 1][j] <= record[i][j] && matrix[i - 1][j] > matrix[i][j]) {
            record[i - 1][j] = record[i][j] + 1;
            max = Math.max(max, record[i - 1][j]);
            depthSearch(i - 1, j, matrix, record);
        }
    }

    /**
     * Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.
     * <p>
     * For example,
     * Given [3,2,1,5,6,4] and k = 2, return 5.
     * <p>
     * Note:
     * You may assume k is always valid, 1 ≤ k ≤ array's length.
     *
     * @param nums
     * @param k
     * @return
     */
    public int findKthLargest1(int[] nums, int k) {
        Arrays.sort(nums);
        return nums[nums.length - k];
    }

    private void shuffle(int a[]) {

        final Random random = new Random();
        for (int ind = 1; ind < a.length; ind++) {
            final int r = random.nextInt(ind + 1);
            exch(a, ind, r);
        }
    }

    public int findKthLargest(int[] nums, int k) {
        shuffle(nums);
        k = nums.length - k;
        int lo = 0;
        int hi = nums.length - 1;
        while (lo < hi) {
            final int j = partition(nums, lo, hi);
            if (j < k) {
                lo = j + 1;
            } else if (j > k) {
                hi = j - 1;
            } else {
                break;
            }
        }
        return nums[k];
    }

    private int partition(int[] a, int lo, int hi) {

        int i = lo;
        int j = hi + 1;
        while (true) {
            while (i < hi && less(a[++i], a[lo])) ;
            while (j > lo && less(a[lo], a[--j])) ;
            if (i >= j) {
                break;
            }
            exch(a, i, j);
        }
        exch(a, lo, j);
        return j;
    }

    private void exch(int[] a, int i, int j) {
        final int tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }

    private boolean less(int v, int w) {
        return v < w;
    }

    /**
     * Write a program to find the nth super ugly number.
     * <p>
     * Super ugly numbers are positive numbers whose all prime factors are in the given prime list primes of size k. For example, [1, 2, 4, 7, 8, 13, 14, 16, 19, 26, 28, 32] is the sequence of the first 12 super ugly numbers given primes = [2, 7, 13, 19] of size 4.
     * <p>
     * Note:
     * (1) 1 is a super ugly number for any given primes.
     * (2) The given numbers in primes are in ascending order.
     * (3) 0 < k ≤ 100, 0 < n ≤ 106, 0 < primes[i] < 1000.
     *
     * @param n
     * @param primes
     * @return
     */
    public int nthSuperUglyNumber(int n, int[] primes) {
        int len = primes.length;
        int[] index = new int[len]; //index[0]==0, ... index[len-1]==0
        int[] res = new int[n];
        res[0] = 1;
        for (int i = 1; i < n; i++) {
            int min = Integer.MAX_VALUE;
            for (int j = 0; j < len; j++) {
                min = Math.min(res[index[j]] * primes[j], min);
            }
            res[i] = min;
            for (int j = 0; j < len; j++) {
                if (res[i] % primes[j] == 0) index[j]++;
            }

        }

        return res[n - 1];
    }


    /**
     * Given a binary tree, flatten it to a linked list in-place.
     * <p>
     * For example,
     * Given
     * <p>
     * 1
     * / \
     * 2   5
     * / \   \
     * 3   4   6
     * The flattened tree should look like:
     * 1
     * \
     * 2
     * \
     * 3
     * \
     * 4
     * \
     * 5
     * \
     * 6
     *
     * @param root
     */
    public void flatten(TreeNode root) {
        if (root != null) {
            flattenHelper(root);
        }
    }


    public void flattern(TreeNode root) {
        if (root == null) return;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        TreeNode pre = null;
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            if (node.left != null) {
                stack.push(node.left);
            }
            if (node.right != null) {
                stack.push(node.right);
            }
            if (pre == null) pre = node;
            else {
                pre.right = node;
                pre.left = null;
            }
        }
    }

    public TreeNode flattenHelper(TreeNode root) {
        if (root.left == null && root.right == null) {
            return root;
        }
        TreeNode ln = null, rn = null;
        if (root.left != null) ln = flattenHelper(root.left);
        if (root.right != null) rn = flattenHelper(root.right);
        if (ln != null) {
            ln.right = root.right;
            root.right = root.left;
            root.left = null;
        }
        return rn == null ? ln : rn;
    }

    /**
     * Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.
     * <p>
     * For example, given n = 12, return 3 because 12 = 4 + 4 + 4; given n = 13, return 2 because 13 = 4 + 9.
     *
     * @param n
     * @return
     */
    public int numSquares(int n) {
        int[] record = new int[n + 1];
        for (int i = 0; i < n + 1; i++) {
            record[i] = Integer.MAX_VALUE;
        }
        return numSquares(n, record);
    }

    public int numSquares(int n, int[] record) {
        if (n == 0) return 0;
        if (record[n] < Integer.MAX_VALUE) return record[n];
        int maxSqrt = (int) Math.sqrt(n);
        for (int i = maxSqrt; i > 0; i--) {
            int rest = (int) (n - Math.pow(i, 2));
            record[n] = Math.min(record[n], 1 + numSquares(rest, record));
        }
        return record[n];
    }

    /**
     * Given a set of distinct integers, nums, return all possible subsets.
     * <p>
     * Note:
     * Elements in a subset must be in non-descending order.
     * The solution set must not contain duplicate subsets.
     * For example,
     * If nums = [1,2,3], a solution is:
     * <p>
     * [
     * [3],
     * [1],
     * [2],
     * [1,2,3],
     * [1,3],
     * [2,3],
     * [1,2],
     * []
     * ]
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsets(int[] nums) {
        if (nums == null || nums.length == 0) {
            return null;
        }
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        subsetsHelper(nums, nums.length - 1, result);
        result.add(new ArrayList<>());
        return result;
    }

    public void subsetsHelper(int[] nums, int i, List<List<Integer>> result) {
        if (i > 0) {
            subsetsHelper(nums, i - 1, result);
        }
        List<List<Integer>> tempResult = new ArrayList<>(result);
        for (List<Integer> subset : tempResult) {
            List<Integer> temp = new ArrayList<>(subset);
            temp.add(nums[i]);
            result.add(temp);
        }
        List<Integer> subset = new ArrayList<>();
        subset.add(nums[i]);
        result.add(subset);
    }

    /**
     * Given a set of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.
     * <p>
     * The same repeated number may be chosen from C unlimited number of times.
     * <p>
     * Note:
     * All numbers (including target) will be positive integers.
     * Elements in a combination (a1, a2, … , ak) must be in non-descending order. (ie, a1 ≤ a2 ≤ … ≤ ak).
     * The solution set must not contain duplicate combinations.
     * For example, given candidate set 2,3,6,7 and target 7,
     * A solution set is:
     * [7]
     * [2, 2, 3]
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < candidates.length; i++) {
            List<Integer> group = new ArrayList<>();
            group.add(candidates[i]);
            combinationSumHelper(result, group, candidates, target - candidates[i]);
        }
        return result;
    }


    public void combinationSumHelper(List<List<Integer>> result, List<Integer> group, int[] candidates, int target) {
        if (target == 0) {
            result.add(group);
            return;
        }
        for (int j = 0; j < candidates.length; j++) {
            List<Integer> tempGroup = new ArrayList<>(group);
            if (candidates[j] == target) {
                if (tempGroup.get(tempGroup.size() - 1) <= candidates[j]) {
                    tempGroup.add(candidates[j]);
                    result.add(tempGroup);
                }
            } else if (candidates[j] < target) {
                if (tempGroup.get(tempGroup.size() - 1) <= candidates[j]) {
                    tempGroup.add(candidates[j]);
                    combinationSumHelper(result, tempGroup, candidates, target - candidates[j]);
                }
            }
        }
    }

    /**
     * Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.
     *
     * @param head
     * @return
     */
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null)
            return null;
        ListNode fast = head;
        ListNode slow = head;
        ListNode prev = null;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            prev = slow;
            slow = slow.next;
        }
        TreeNode root = new TreeNode(slow.val);
        if (prev != null)
            prev.next = null;
        else
            head = null;

        root.left = sortedListToBST(head);
        root.right = sortedListToBST(slow.next);
        return root;
    }


    /**
     * Given a collection of integers that might contain duplicates, nums, return all possible subsets.
     * <p>
     * Note:
     * Elements in a subset must be in non-descending order.
     * The solution set must not contain duplicate subsets.
     * For example,
     * If nums = [1,2,2], a solution is:
     * <p>
     * [
     * [2],
     * [1],
     * [1,2,2],
     * [2,2],
     * [1,2],
     * []
     * ]
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> empty = new ArrayList<>();
        result.add(empty);
        Arrays.sort(nums);

        for (int i = 0; i < nums.length; i++) {
            int dupCount = 0;
            while (((i + 1) < nums.length) && nums[i + 1] == nums[i]) {
                dupCount++;
                i++;
            }
            int prevNum = result.size();
            for (int j = 0; j < prevNum; j++) {
                List<Integer> element = new ArrayList<>(result.get(j));
                for (int t = 0; t <= dupCount; t++) {
                    element.add(nums[i]);
                    result.add(new ArrayList<>(element));
                }
            }
        }
        return result;
    }

    public ArrayList<ArrayList<Integer>> subsetsWithDupRecursive(ArrayList<Integer> S) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (S == null || S.size() == 0) return result;
        Collections.sort(S);
        dfs(result, S, 0);
        ArrayList<Integer> empty = new ArrayList<>();
        result.add(empty);
        return result;
    }

    void dfs(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> S, int index) {
        if (index == S.size()) return;
        int initIndex = result.size();
        ArrayList<ArrayList<Integer>> copy = new ArrayList<>(result);
        for (ArrayList<Integer> item : copy) {
            ArrayList<Integer> cp = new ArrayList<>(item);
            cp.add(S.get(index));
            result.add(cp);
        }
        ArrayList<Integer> item = new ArrayList<>();
        item.add(S.get(index));
        result.add(item);

        while (index < S.size() - 1 && S.get(index).equals(S.get(index + 1))) {
            int i = initIndex;
            ArrayList<ArrayList<Integer>> copy1 = new ArrayList<>(result);
            for (; i < copy1.size(); i++) {
                ArrayList<Integer> temp = new ArrayList<>(copy1.get(i));
                temp.add(S.get(index));
                result.add(temp);
            }
            initIndex = i;
            index++;
        }
        dfs(result, S, index + 1);
    }

    public int rob(int[] nums) {
        if (nums.length == 1) return nums[0];
        return Math.max(rob(nums, 0, nums.length - 2), rob(nums, 1, nums.length - 1));
    }

    private int rob(int[] num, int lo, int hi) {
        int include = 0, exclude = 0;
        for (int j = lo; j <= hi; j++) {
            int i = include, e = exclude;
            include = e + num[j];
            exclude = Math.max(e, i);
        }
        return Math.max(include, exclude);
    }

    /**
     * Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.
     * <p>
     * For example, given the following triangle
     * [
     * [2],
     * [3,4],
     * [6,5,7],
     * [4,1,8,3]
     * ]
     * The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
     *
     * @param triangle
     * @return
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle == null || triangle.size() == 0) return 0;
        if (triangle.size() == 1) return triangle.get(0).get(0);
        int i = 1;
        while (i < triangle.size()) {
            int j = 0;
            while (j < triangle.get(i).size()) {
                triangle.get(i).set(j, minFromUp(i, j, triangle));
                ++j;
            }
            ++i;
        }
        int min = Integer.MAX_VALUE;
        for (Integer integer : triangle.get(triangle.size() - 1)) {
            min = Math.min(min, integer);
        }
        return min;
    }

    private Integer minFromUp(int i, int j, List<List<Integer>> triangle) {
        int cur = triangle.get(i).get(j);
        if (j == 0) {
            return triangle.get(i - 1).get(j) + cur;
        }
        if (j == triangle.get(i).size() - 1) {
            return triangle.get(i - 1).get(j - 1) + cur;
        } else {
            int upLeft = triangle.get(i - 1).get(j - 1);
            int upRight = triangle.get(i - 1).get(j);
            return Math.min(upLeft + cur, upRight + cur);
        }
    }

    /**
     * Given a range [m, n] where 0 <= m <= n <= 2147483647, return the bitwise AND of all numbers in this range, inclusive.
     * <p>
     * For example, given the range [5, 7], you should return 4.
     *
     * @param m
     * @param n
     * @return
     */
    public int rangeBitwiseAnd(int m, int n) {
        if (m == 0) return 0;
        int moveFac = 1;
        while (m != n) {
            m >>= 1;
            n >>= 1;
            moveFac <<= 1;
        }
        return m * moveFac;
    }

    /**
     * Follow up for "Unique Paths":
     * <p>
     * Now consider if some obstacles are added to the grids. How many unique paths would there be?
     * <p>
     * An obstacle and empty space is marked as 1 and 0 respectively in the grid.
     * <p>
     * For example,
     * There is one obstacle in the middle of a 3x3 grid as illustrated below.
     * <p>
     * [
     * [0,0,0],
     * [0,1,0],
     * [0,0,0]
     * ]
     * The total number of unique paths is 2.
     * <p>
     * Note: m and n will be at most 100.
     *
     * @param obstacleGrid
     * @return
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0 || obstacleGrid[0][0] == 1) return 0;
        int m = obstacleGrid.length, n = obstacleGrid[0].length;
        obstacleGrid[0][0] = 1;
        for (int i = 1; i < n; i++) {
            if (obstacleGrid[0][i] == 1) {
                for (int j = i; j < n; j++) {
                    obstacleGrid[0][j] = 0;
                }
                break;
            } else {
                obstacleGrid[0][i] = 1;
            }
        }
        for (int i = 1; i < m; i++) {
            if (obstacleGrid[i][0] == 1) {
                for (int j = i; j < m; j++) {
                    obstacleGrid[j][0] = 0;
                }
                break;
            } else {
                obstacleGrid[i][0] = 1;
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    obstacleGrid[i][j] = 0;
                    continue;
                } else {
                    if (obstacleGrid[i - 1][j] == -1 && obstacleGrid[i][j - 1] == -1) {
                        obstacleGrid[i][j] = 0;
                    } else if (obstacleGrid[i - 1][j] == -1) {
                        obstacleGrid[i][j] = obstacleGrid[i][j - 1];
                    } else if (obstacleGrid[i][j - 1] == -1) {
                        obstacleGrid[i][j] = obstacleGrid[i - 1][j];
                    } else {
                        obstacleGrid[i][j] = obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1];
                    }
                }
            }
        }
        return obstacleGrid[m - 1][n - 1];
    }

    public List<TreeNode> generateTrees(int n) {
        return genTreeList(1, n);
    }

    private List<TreeNode> genTreeList(int start, int end) {
        List<TreeNode> list = new ArrayList<>();
        if (start > end) {
            list.add(null);
            return list;
        }
        for (int idx = start; idx <= end; idx++) {
            List<TreeNode> leftList = genTreeList(start, idx - 1);
            List<TreeNode> rightList = genTreeList(idx + 1, end);
            for (TreeNode left : leftList) {
                for (TreeNode right : rightList) {
                    TreeNode root = new TreeNode(idx);
                    root.left = left;
                    root.right = right;
                    list.add(root);
                }
            }
        }
        return list;
    }

    /**
     * One way to serialize a binary tree is to use pre-order traversal. When we encounter a non-null node, we record the node's value. If it is a null node, we record using a sentinel value such as #.
     * <p>
     * _9_
     * /   \
     * 3     2
     * / \   / \
     * 4   1  #  6
     * / \ / \   / \
     * # # # #   # #
     * For example, the above binary tree can be serialized to the string "9,3,4,#,#,1,#,#,2,#,6,#,#", where # represents a null node.
     * <p>
     * Given a string of comma separated values, verify whether it is a correct preorder traversal serialization of a binary tree. Find an algorithm without reconstructing the tree.
     * <p>
     * Each comma separated value in the string must be either an integer or a character '#' representing null pointer.
     * <p>
     * You may assume that the input format is always valid, for example it could never contain two consecutive commas such as "1,,3".
     * <p>
     * Example 1:
     * "9,3,4,#,#,1,#,#,2,#,6,#,#"
     * Return true
     * <p>
     * Example 2:
     * "1,#"
     * Return false
     * <p>
     * Example 3:
     * "9,#,#,1"
     * Return false
     *
     * @param preorder
     * @return
     */
    public boolean isValidSerialization(String preorder) {
        Stack<String> stack = new Stack<>();
        if (preorder.equals("#")) return true;
        return serHelper(preorder, stack);
    }


    public boolean serHelper(String preorder, Stack<String> stack) {
        String[] orders = preorder.split(",");
        for (int i = 0; i < orders.length; i++) {
            if (isNumber(orders[i])) {
                if (stack.isEmpty()) {
                    if (i > 0) {
                        return false;
                    }
                    stack.push("#");
                    stack.push("#");
                } else {
                    stack.pop();
                    stack.push("#");
                    stack.push("#");
                }
            } else {
                if (stack.isEmpty()) {
                    return false;
                } else {
                    stack.pop();
                }
            }
        }
        return stack.isEmpty();
    }

    public boolean isNumber(String c) {
        try {
            Integer.valueOf(c);
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    /**
     * Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
     * <p>
     * You should preserve the original relative order of the nodes in each of the two partitions.
     * <p>
     * For example,
     * Given 1->4->3->2->5->2 and x = 3,
     * return 1->2->2->4->3->5.
     *
     * @param head
     * @param x
     * @return
     */
    public ListNode partition(ListNode head, int x) {
        ListNode dm1 = new ListNode(0);
        ListNode dm2 = new ListNode(0);
        ListNode cur1 = dm1, cur2 = dm2;
        while (head != null) {
            if (head.val < x) {
                cur1.next = head;
                cur1 = head;
            } else {
                cur2.next = head;
                cur2 = head;
            }
            head = head.next;
        }
        cur2.next = null;
        cur1.next = dm2.next;
        return dm1.next;
    }

    public ListNode insertionSortList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode node = new ListNode(head.val);
        ListNode cursor = head.next;
        while (cursor != null) {
            if (cursor.val < node.val) {
                ListNode temp = new ListNode(cursor.val);
                temp.next = node;
                node = temp;
            } else {
                ListNode cur = node, temp = new ListNode(cursor.val);
                while (cur.next != null && cur.next.val <= cursor.val) {
                    cur = cur.next;
                }
                temp.next = cur.next;
                cur.next = temp;
            }
            cursor = cursor.next;
        }
        return node;
    }

    /**
     * Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
     * <p>
     * According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes v and w as the lowest node in T that has both v and w as descendants (where we allow a node to be a descendant of itself).”
     * <p>
     * _______3______
     * /              \
     * ___5__          ___1__
     * /      \        /      \
     * 6      _2       0       8
     * /  \
     * 7   4
     * For example, the lowest common ancestor (LCA) of nodes 5 and 1 is 3. Another example is LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        return left == null ? right : right == null ? left : root;
    }

    int pInorder;   // index of inorder array
    int pPostorder; // index of postorder array

    private TreeNode buildTree(int[] inorder, int[] postorder, TreeNode end) {
        if (pPostorder < 0) {
            return null;
        }

        // create root node
        TreeNode n = new TreeNode(postorder[pPostorder--]);

        // if right node exist, create right subtree
        if (inorder[pInorder] != n.val) {
            n.right = buildTree(inorder, postorder, n);
        }

        pInorder--;

        // if left node exist, create left subtree
        if ((end == null) || (inorder[pInorder] != end.val)) {
            n.left = buildTree(inorder, postorder, end);
        }

        return n;
    }

    /**
     * Given inorder and postorder traversal of a tree, construct the binary tree.
     * TODO
     *
     * @param inorder
     * @param postorder
     * @return
     */
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        pInorder = inorder.length - 1;
        pPostorder = postorder.length - 1;

        return buildTree(inorder, postorder, null);
    }

    /**
     * Given an array S of n integers, find three integers in S such that the sum is closest to a given number, target. Return the sum of the three integers. You may assume that each input would have exactly one solution.
     * <p>
     * For example, given array S = {-1 2 1 -4}, and target = 1.
     * <p>
     * The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
     *
     * @param nums
     * @param target
     * @return
     */
    public int threeSumClosest(int[] nums, int target) {
        int distance = Integer.MAX_VALUE, finalSum = 0;
        if (nums.length == 0 || nums.length == 1 || nums.length == 2) return 0;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int j = i + 1, k = nums.length - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                int tmpDis = Math.abs(sum - target);
                if (sum > target) {
                    if (tmpDis < distance) {
                        distance = tmpDis;
                        finalSum = sum;
                    }
                    k--;
                } else if (sum < target) {
                    if (tmpDis < distance) {
                        distance = tmpDis;
                        finalSum = sum;
                    }
                    j++;
                } else {
                    return sum;
                }
            }
        }
        return finalSum;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums.length == 0 || nums.length == 1 || nums.length == 2) return result;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int j = i + 1, k = nums.length - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum > 0) {
                    k--;
                    while (k > j && nums[k] == nums[k + 1]) k--;
                } else if (sum < 0) {
                    j++;
                    while (j < k && nums[j] == nums[j - 1]) j++;
                } else {
                    List<Integer> set = new ArrayList<>();
                    set.add(nums[i]);
                    set.add(nums[j]);
                    set.add(nums[k]);
                    result.add(set);
                    j++;
                    k--;
                    while (j < k && nums[j] == nums[j - 1]) j++;
                    while (k > j && nums[k] == nums[k + 1]) k--;
                }
            }
        }
        return result;
    }

    public class NumMatrix {

        int[][] dp;

        public NumMatrix(int[][] matrix) {
            dp = new int[matrix.length + 1][matrix[0].length + 1];
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[0].length; j++) {
                    dp[i + 1][j + 1] = dp[i + 1][j] + dp[i][j + 1] + matrix[i][j] - dp[i][j];
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            return dp[row2 + 1][col2 + 1] - dp[row1][col2 + 1] - dp[row2 + 1][col1] + dp[row1][col1];
        }
    }

    /**
     * Given a string, find the length of the longest substring without repeating characters. For example, the longest substring without repeating letters for "abcabcbb" is "abc", which the length is 3. For "bbbbb" the longest substring is "b", with the length of 1.
     *
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {

        if (s == null || s.isEmpty()) return 0;
        if (s.length() == 1) return 1;
        Map<Character, Integer> table = new HashMap<>();
        int slow = 0, fast = 1, ll = 1;
        table.put(s.charAt(slow), slow);
        while (fast < s.length()) {
            if (table.get(s.charAt(fast)) == null || table.get(s.charAt(fast)) < slow) {
                table.put(s.charAt(fast), fast);
                fast++;
                ll = Math.max(ll, fast - slow);
            } else {
                int dupIdx = table.get(s.charAt(fast));
                slow = dupIdx + 1;
            }
        }
        return ll;
    }

    public String reverseWords(String s) {
        String[] arr = s.trim().split(" ");
        if (arr.length == 0) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = arr.length - 1; i >= 0; i--) {
            if (!arr[i].trim().isEmpty()) {
                sb.append(arr[i].trim());
                sb.append(" ");
            }
        }
        return sb.toString().trim();
    }

    /**
     * Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.
     * <p>
     * Note:
     * Elements in a quadruplet (a,b,c,d) must be in non-descending order. (ie, a ≤ b ≤ c ≤ d)
     * The solution set must not contain duplicate quadruplets.
     * For example, given array S = {1 0 -1 0 -2 2}, and target = 0.
     * <p>
     * A solution set is:
     * (-1,  0, 0, 1)
     * (-2, -1, 1, 2)
     * (-2,  0, 0, 2)
     *
     * @param nums
     * @param target
     * @return
     */
    public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            for (int j = i + 1; j < nums.length; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                twoSum(nums, i, j, (target - nums[i] - nums[j]), result);
            }
        }
        return result;
    }

    void twoSum(int[] nums, int idx1, int idx2, int target, List<List<Integer>> result) {
        int i = idx2 + 1, j = nums.length - 1;
        while (i < j) {
            int temp = nums[i] + nums[j];
            if (temp > target) {
                j--;
            } else if (temp < target) {
                i++;
            } else {
                List<Integer> subset = new ArrayList<>();
                subset.add(nums[idx1]);
                subset.add(nums[idx2]);
                subset.add(nums[i]);
                subset.add(nums[j]);
                result.add(subset);
                i++;
                j--;
                while (i < j && nums[i] == nums[i - 1]) i++;
                while (i < j && nums[j] == nums[j + 1]) j--;
            }
        }
    }

    public List<Integer> majorityElement(int[] nums) {
        if (nums.length == 0) return new ArrayList<>();
        int candi1 = 0, candi2 = 0, count1 = 0, count2 = 0;
        for (int n : nums) {
            if (n == candi1) {
                count1++;
            } else if (n == candi2) {
                count2++;
            } else if (count1 == 0) {
                candi1 = n;
                count1 = 1;
            } else if (count2 == 0) {
                candi2 = n;
                count2 = 1;
            } else {
                count1--;
                count2--;
            }
        }
        int sum1 = 0, sum2 = 0;
        for (int n : nums) {
            if (n == candi1) {
                sum1++;
            }
            if (n == candi2) {
                sum2++;
            }
        }
        List<Integer> list = new ArrayList<>();
        if (sum1 > nums.length / 3) list.add(candi1);
        if (sum2 > nums.length / 3 && candi1 != candi2) list.add(candi2);
        return list;
    }

    /**
     * k out of n combinations
     *
     * @param n
     * @param k
     * @return
     */
    public LinkedList<LinkedList<Integer>> combinations(int n, int k) {
        LinkedList<LinkedList<Integer>> result = new LinkedList<>();
        if (n < k || n <= 0) return result;
        LinkedList<Integer> list = new LinkedList<>();
        DFS(result, list, n, k, 1);
        return result;
    }

    public void DFS(LinkedList<LinkedList<Integer>> result, LinkedList<Integer> curr, int n, int k, int level) {
        if (curr.size() == k) {
            result.offer(new LinkedList<>(curr));
            return;
        }
        if (curr.size() > k) {
            return;
        }
        for (int i = level; i <= n; i++) {
            curr.offer(i);
            DFS(result, curr, n, k, i + 1);
            curr.removeLast();
        }
    }

    public int[] searchRange(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        int[] result = new int[]{-1, -1};
        if (nums.length == 0) return result;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] >= target) {
                right = mid;
            } else {
                left = mid;
            }
        }
        if (left < nums.length && nums[left] == target) {
            result[0] = left;
        } else {
            return result;
        }
        right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] <= target) {
                left = mid;
            } else {
                right = mid;
            }
        }
        result[1] = right;
        return result;
    }

    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        pathDfs(result, new ArrayList<>(), sum, root);
        return result;
    }

    public void pathDfs(List<List<Integer>> result, List<Integer> cur, int sum, TreeNode root) {
        if (root.left == null && root.right == null && root.val == sum) {
            cur.add(root.val);
            result.add(cur);
            return;
        } else {
            if (root.val > sum) return;
            cur.add(root.val);
            if (root.left != null) {
                pathDfs(result, new ArrayList<>(cur), sum - root.val, root.left);
            }
            if (root.right != null) {
                pathDfs(result, new ArrayList<>(cur), sum - root.val, root.right);
            }
        }
    }

    public boolean isValidBST(TreeNode root) {
        return isValid(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    public boolean isValid(TreeNode node, long min, long max) {
        if (node == null) return true;
        if (node.val >= max || node.val <= min) return false;
        return isValid(node.left, min, node.val) && isValid(node.right, node.val, max);
    }

    public boolean verifyPreorder(int[] preorder) {
        if (preorder == null) return true;
        Stack<Integer> stack = new Stack<>();
        int low = Integer.MIN_VALUE;
        for (int p : preorder) {
            if (p < low) return false;
            while (!stack.isEmpty() && p > stack.peek()) {
                low = stack.pop();
            }
            stack.push(p);
        }
        return true;
    }

    public int maxProduct(int[] nums) {
        /*
        maxDP[i+1]=max(maxDP[i]*nums[i+1],maxDP[i],minDP[i]*nums[i+1]
        minDP[i+1]=min(minDP[i]*nums[i+1],minDP[i],maxDP[i]*nums[i+1]
         */
        if (nums.length == 0) {
            return 0;
        }
        int maxherepre = nums[0];
        int minherepre = nums[0];
        int maxsofar = nums[0];
        int maxhere, minhere;

        for (int i = 1; i < nums.length; i++) {
            maxhere = Math.max(Math.max(maxherepre * nums[i], minherepre * nums[i]), nums[i]);
            minhere = Math.min(Math.min(maxherepre * nums[i], minherepre * nums[i]), nums[i]);
            maxsofar = Math.max(maxhere, maxsofar);
            maxherepre = maxhere;
            minherepre = minhere;
        }
        return maxsofar;
    }

    public int triangle(int[][] triangle) {
        int row = triangle.length;
        int[] table = new int[triangle[row - 1].length];
        for (int i = 0; i < table.length; i++) {
            table[i] = triangle[row - 1][i];
        }
        for (int i = row - 2; i >= 0; i--) {
            for (int j = 0; j <= i; j++) {
                table[j] = triangle[i][j] + Math.min(table[j], table[j + 1]);
            }
        }
        return table[0];
    }

    int numSquaresDP(int n) {
        if (n == 0 || n == 1) return 1;
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 1; i <= n; i++) {
            int sqrt = (int) Math.sqrt(i);
            for (int j = 1; j <= sqrt; j++) {
                dp[i] = Math.min(dp[i], Math.min(dp[i - 1] + 1, dp[i - j * j] + 1));
            }
        }
        return dp[n];
    }

    public int canCompleteCircuit(int[] gas, int[] cost) {
        int gasSum = 0, costSum = 0;
        for (int ga : gas) {
            gasSum += ga;
        }
        for (int i : cost) {
            costSum += i;
        }
        if (gasSum < costSum) return -1;
        else {
            int i = 0, rest = 0, s = 0;
            while (i < gas.length) {
                rest += gas[i] - cost[i];
                if (rest < 0) {
                    s = i + 1;
                    rest = 0;
                }
                i++;
            }
            return s;
        }
    }


    public boolean canJump(int[] nums) {
        if (nums == null || nums.length == 0) return true;
        int step = nums[0];
        if (step < 1 && nums.length > 1) return false;
        for (int i = 1; i < nums.length; i++) {
            step--;
            if (step < 0) return false;
            step = Math.max(step, nums[i]);
        }
        return true;
    }

    public boolean wordBreak(String s, Set<String> wordDict) {
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (dp[j] && wordDict.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int promo = 0;
        ListNode dummy = new ListNode(0);
        ListNode node = dummy;
        while (l1 != null && l2 != null) {
            int sum = l1.val + l2.val + promo;
            promo = sum / 10;
            node.next = new ListNode(sum % 10);
            node = node.next;
            l1 = l1.next;
            l2 = l2.next;
        }
        while (l1 != null) {
            int sum = promo + l1.val;
            promo = sum / 10;
            node.next = new ListNode(sum % 10);
            node = node.next;
            l1 = l1.next;
        }
        while (l2 != null) {
            int sum = promo + l2.val;
            promo = sum / 10;
            node.next = new ListNode(sum % 10);
            node = node.next;
            l2 = l2.next;
        }
        if (promo == 1) {
            node.next = new ListNode(1);
        }
        return dummy.next;
    }

    public ListNode reverseBetween(ListNode head, int m, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode p = dummy;
        for (int i = 1; i < m; i++) {
            p = p.next;
        }
        ListNode cur = p.next;
        while (m < n) {
            ListNode next = cur.next;
            cur.next = next.next;
            next.next = p.next;
            p.next = next;
            m++;
        }
        return dummy.next;
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        if (k <= 1 || head == null) return head;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode p = dummy;
        ListNode pre = dummy;
        while (p != null) {
            pre = p;
            for (int i = 0; i < k; i++) {
                p = p.next;
                if (p == null) {
                    return dummy.next;
                }
            }
            p = reverseBetween(pre, p);
        }
        return dummy.next;
    }

    private ListNode reverseBetween(ListNode pre, ListNode end) {
        ListNode cur = pre.next;
        while (pre.next != end) {
            ListNode next = cur.next;
            cur.next = next.next;
            next.next = pre.next;
            pre.next = next;
        }
        return cur;
    }

    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        return sortListHelper(head);
    }

    public ListNode sortListHelper(ListNode left) {
        if (left == null || left.next == null) {
            return left;
        }
        ListNode slow = left;
        ListNode fast = left;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        fast = slow.next;
        slow.next = null;
        ListNode l1 = sortListHelper(left);
        ListNode l2 = sortListHelper(fast);
        return merge(l1, l2);
    }

    private ListNode merge(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode p = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                p.next = l1;
                l1 = l1.next;
            } else {
                p.next = l2;
                l2 = l2.next;
            }
            p = p.next;
        }
        if (l1 != null) {
            p.next = l1;
        } else {
            p.next = l2;
        }
        return dummy.next;
    }

    public ListNode rotateRight(ListNode head, int k) {
        if (k == 0 || head == null) return head;
        int n = 0;
        ListNode p = head;
        ListNode pre = null;
        while (p != null) {
            n++;
            pre = p;
            p = p.next;
        }
        int steps = k % n;
        if (steps == 0) {
            return head;
        }
        pre.next = head;
        for (int i = 0; i < n - steps; i++) {
            pre = pre.next;
        }
        ListNode h = pre.next;
        pre.next = null;
        return h;
    }

    public void reorderList(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) return;
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode right = slow.next;
        slow.next = null;
        fast = reverse(right);
        ListNode p = head;
        while (p != null && fast != null) {
            ListNode leftNext = p.next;
            ListNode rightNext = fast.next;
            p.next = fast;
            fast.next = leftNext;
            p = leftNext;
            fast = rightNext;
        }
    }

//    private ListNode reverse(ListNode right) {
//        ListNode pre = null;
//        ListNode p = right;
//        while (p != null) {
//            ListNode next = p.next;
//            p.next = pre;
//            pre = p;
//            p = next;
//        }
//        return pre;
//    }

    /**
     * Boyer–Moore string search algorithm
     *
     * @param source
     * @param target
     * @return
     */
    public int strStr(String source, String target) {
        if (source == null || target == null) return -1;
        if (target.length() > source.length()) return -1;
        if (target.length() == 0 || source.length() == 0) return 0;
        Map<Character, Integer> badMatchTable = new HashMap<>();
        for (int i = 0; i < target.length(); i++) {
            badMatchTable.put(target.charAt(i), Math.max(1, target.length() - 1 - i));
        }
        badMatchTable.put(target.charAt(target.length() - 1), target.length());
        int i = target.length() - 1;
        while (i < source.length()) {
            int j = target.length() - 1, k = i;
            while (j >= 0 && source.charAt(k) == target.charAt(j)) {
                j--;
                k--;
            }
            if (j == -1) return k + 1;
            else
                i += badMatchTable.get(source.charAt(i)) == null ? target.length() : badMatchTable.get(source.charAt(i));
        }
        return -1;
    }

    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> result = new LinkedList<>();
        if (nums == null || nums.length == 0) return result;
        List<Integer> cur = new LinkedList<>();
        boolean[] visited = new boolean[nums.length];
        Arrays.sort(nums);

        permuteUniqueHelper(result, cur, nums, visited);
        return result;
    }

    public void permuteUniqueHelper(List<List<Integer>> result, List<Integer> cur, int[] nums, boolean[] visited) {
        if (cur.size() == nums.length) {
            result.add(new LinkedList<>(cur));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i]) continue;
            if (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1]) continue;
            cur.add(nums[i]);
            visited[i] = true;
            permuteUniqueHelper(result, cur, nums, visited);
            visited[i] = false;
            cur.remove(cur.size() - 1);
        }
    }

    public int threeSumSmaller(int[] nums, int target) {
        Arrays.sort(nums);
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += twoSumSmaller(i + 1, nums, target - nums[i]);
        }
        return sum;
    }

    int twoSumSmaller(int start, int[] nums, int target) {
        int index = 0;
        for (int i = nums.length - 1; i > start; i--) {
            if (nums[i] + nums[start] < target) {
                index = i;
                break;
            }
        }
        return index == 0 ? 0 : index - start + 1;
    }

    public ListNode addLists2(ListNode l1, ListNode l2) {
        // write your code here
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        ListNode r1 = reverse(l1);
        ListNode r2 = reverse(l2);
        int carry = 0;
        ListNode dummy = new ListNode(0);
        ListNode dm = dummy;
        while (r1 != null && r2 != null) {
            int val = r1.val + r2.val + carry;
            dm.next = new ListNode(val % 10);
            carry = (val >= 10) ? 1 : 0;
            dm = dm.next;
            r1 = r1.next;
            r2 = r2.next;
        }
        while (r1 != null) {
            int val = r1.val + carry;
            dm.next = new ListNode(val % 10);
            carry = (val >= 10) ? 1 : 0;
            dm = dm.next;
            r1 = r1.next;
        }
        while (r2 != null) {
            int val = r2.val + carry;
            dm.next = new ListNode(val % 10);
            carry = (val >= 10) ? 1 : 0;
            dm = dm.next;
            r2 = r2.next;
        }
        if (carry > 0) {
            dm.next = new ListNode(carry);
        }
        return reverse(dummy.next);
    }

    ListNode reverse(ListNode n) {
        ListNode pre = null;
        ListNode node = n;
        while (node != null) {
            ListNode next = node.next;
            node.next = pre;
            pre = node;
            node = next;
        }
        return pre;
    }

    public int maxDiffSubArrays(int[] nums) {
        // write your code here
        int[] leftMax = new int[nums.length];
        int[] rightMin = new int[nums.length];
        int sum = 0, minSum = 0, max = Integer.MIN_VALUE;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            max = leftMax[i] = Math.max(max, sum - minSum);
            minSum = Math.min(minSum, sum);
        }

        sum = 0;
        int maxSum = 0, min = Integer.MAX_VALUE;
        for (int i = nums.length - 1; i >= 0; i--) {
            sum += nums[i];
            min = rightMin[i] = Math.min(min, sum - maxSum);
            maxSum = Math.max(maxSum, sum);
        }
        int maxDiff = Integer.MIN_VALUE;
        for (int i = 0; i < nums.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(leftMax[i] - rightMin[i]));
        }
        return maxDiff;
    }

    public int maxSubArray(int[] nums, int k) {
        // write your code here
        if (nums == null || nums.length < k) {
            return 0;
        }
        int n = nums.length;
        int[][][] dp = new int[n + 1][k + 1][2];
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= k; j++) {
                if (j > i) {
                    dp[i][j][0] = Integer.MIN_VALUE;
                    dp[i][j][1] = Integer.MIN_VALUE;
                }
            }
        }

        dp[0][0][0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= k; j++) {
                if (j > i) continue;
                dp[i][j][0] = Math.max(0, Math.max(dp[i - 1][j][0], dp[i - 1][j][1]));
                dp[i][j][1] = Math.max(dp[i - 1][j - 1][0], Math.max(dp[i - 1][j][1], dp[i - 1][j - 1][0])) + nums[i - 1];
            }
        }
        return Math.max(dp[n][k][0], dp[n][k][1]);
    }


    public static void main(String[] args) {
        new Medium().longestConsecutive(new int[]{100,4,200,1,3,2});
    }

    public int longestConsecutive(int[] num) {
        // write you code here
        heapity(num);
        int local = 1;
        int global = 1;
        for(int i = 1;i < num.length;i++){
            if(num[i] - num[i - 1] == 1){
                local++;
            }else if(num[i] == num[i - 1]){
                //nothing
            }else{
                global = Math.max(global, local);
                local = 1;
            }
        }
        global = Math.max(global,local);
        return global;
    }


    void heapity(int[] A){
        for(int i = A.length / 2;i >= 0;i--){
            int k = i;
            while(k < A.length){
                int smallest = k;
                int left = 2 * k + 1;
                int right = 2 * k + 2;
                if(left < A.length && A[left] < A[smallest]){
                    smallest = left;
                }
                if(right < A.length && A[right] < A[smallest]){
                    smallest = right;
                }
                if(smallest == k) break;
                swap(A, smallest, k);
//                int temp = A[smallest];
//                A[smallest] = A[k];
//                A[k] = temp;
                k = smallest;
            }
        }
    }

    class Node{
        String val;
        List<Node> neighbors;
        int distance;
        public Node(String v){
            this.val = v;
            this.neighbors = new ArrayList<>();
            this.distance = 0;
        }
    }
    /**
     * @param start, a string
     * @param end, a string
     * @param dict, a set of string
     * @return a list of lists of string
     */
    public List<List<String>> findLadders(String start, String end, Set<String> dict) {
        // write your code here
        Node endNode = new Node(end);
        Map<String,Node> map = new HashMap<>();
        map.put(end,endNode);
        Queue<Node> queue = new LinkedList<>();
        queue.offer(endNode);
        while(!queue.isEmpty()){
            Node node = queue.poll();
            String val = node.val;
            for(int i = 0;i < val.length();i++){
                for(char c = 'a';c <= 'z';c++){
                    String sub = val.substring(0,i) + c + val.substring(i + 1);
                    if(!dict.contains(sub)){
                        continue;
                    }
                    if(!map.containsKey(sub)){
                        Node n = new Node(sub);
                        n.distance = node.distance + 1;
                        map.put(sub,n);
                        queue.offer(n);
                    }
                    map.get(sub).neighbors.add(node);
                }
            }
        }
        Node startNode = map.get(start);
        List<List<String>> result = new ArrayList<>();
        List<String> cur = new ArrayList<>();
        cur.add(start);
        dfs(startNode, cur, result);
        return result;
    }

    void dfs(Node start, List<String> cur, List<List<String>> result){
        if(start.distance == 0){
            result.add(new ArrayList<>(cur));
        }else{
            for(Node neighbor : start.neighbors){
                if(neighbor.distance == start.distance - 1){
                    cur.add(neighbor.val);
                    dfs(neighbor, cur, result);
                    cur.remove(cur.size() - 1);
                }
            }
        }
    }



    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        // write your code here
        if (node == null) {
            return null;
        } else {
            Map<UndirectedGraphNode, UndirectedGraphNode> map = new HashMap<>();
            Queue<UndirectedGraphNode> q = new LinkedList<>();
            q.offer(node);
            while (!q.isEmpty()) {
                UndirectedGraphNode n = q.poll();
                UndirectedGraphNode newNode = new UndirectedGraphNode(n.label);
                map.put(n, newNode);
                for (UndirectedGraphNode nei : n.neighbors) {
                    if (!map.containsKey(nei)) {
                        q.offer(nei);
                    }
                }
            }
            q.offer(node);
            Set<UndirectedGraphNode> set = new HashSet<>();
            while (!q.isEmpty()) {
                UndirectedGraphNode n = q.poll();
                UndirectedGraphNode newNode = map.get(n);
                for (UndirectedGraphNode nei : n.neighbors) {
                    newNode.neighbors.add(map.get(nei));
                    if (!set.contains(nei)) {
                        q.offer(nei);
                    }
                }
                set.add(n);
            }
            return map.get(node);
        }
    }

    static class UndirectedGraphNode {
        int label;
        ArrayList<UndirectedGraphNode> neighbors;

        public UndirectedGraphNode(int x) {
            label = x;
            neighbors = new ArrayList<UndirectedGraphNode>();
        }
    }
}
