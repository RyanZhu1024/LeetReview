package leet;

import java.util.*;

/**
 * Created by RyanZhu on 11/13/15.
 */
public class Easy {
    /**
     * You are playing the following Nim Game with your friend: There is a heap of stones on the table, each time one of you take turns to remove 1 to 3 stones. The one who removes the last stone will be the winner. You will take the first turn to remove the stones.
     * <p>
     * Both of you are very clever and have optimal strategies for the game. Write a function to determine whether you can win the game given the number of stones in the heap.
     * <p>
     * For example, if there are 4 stones in the heap, then you will never win the game: no matter 1, 2, or 3 stones you remove, the last stone will always be removed by your friend.
     *
     * @param n
     * @return
     */
    public boolean nimGame(int n) {
        //because your opponent's strategy is to take away the number of stones that just make this round 4.
        return n % 4 != 0;
    }

    /**
     * Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.
     * <p>
     * For example:
     * <p>
     * Given num = 38, the process is like: 3 + 8 = 11, 1 + 1 = 2. Since 2 has only one digit, return it.
     * <p>
     * Follow up:
     * Could you do it without any loop/recursion in O(1) runtime?
     *
     * @param num
     * @return
     */
    public int addDigits(int num) {
        int sum = 0;
        while (num > 0) {
            sum += num % 10;
            num /= 10;
        }
        if (sum > 9) {
            return addDigits(sum);
        } else {
            return sum;
        }
    }

    public int addDigitsInO1(int num) {
        if (num == 0) {
            return num;
        } else {
            return 1 + (num - 1) % 9;
        }
    }


    // Definition for a binary tree node.
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }


    /**
     * binary tree max depth
     *
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root) {
//        return root==null?0:Math.max(maxDepth(root.left),maxDepth(root.right))+1;
        if (root == null) {
            return 0;
        }
        int leftMax = maxDepth(root.left);
        int rightMax = maxDepth(root.right);
        return leftMax > rightMax ? leftMax + 1 : rightMax + 1;
    }

    //     Definition for singly-linked list.
    public class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }

    }

    /**
     * Write a function to delete a node (except the tail) in a singly linked list, given only access to that node.
     * <p>
     * Supposed the linked list is 1 -> 2 -> 3 -> 4 and you are given the third node with value 3, the linked list should become 1 -> 2 -> 4 after calling your function.
     *
     * @param node
     */
    public void deleteNode(ListNode node) {
        if (node.next != null) {
            node.val = node.next.val;
            node.next = node.next.next;
        }
    }

    /**
     * Given two binary trees, write a function to check if they are equal or not.
     * <p>
     * Two binary trees are considered equal if they are structurally identical and the nodes have the same value.
     *
     * @param p
     * @param q
     * @return
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if ((p == null && q != null) || (p != null && q == null)) {
            return false;
        }
        if (p == null && q == null) {
            return true;
        }
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right) && p.val == q.val;
    }

    /**
     * Given an array of integers, find if the array contains any duplicates. Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.
     *
     * @param nums
     * @return
     */
    public boolean containsDuplicate(int[] nums) {
        if (nums == null || nums.length <= 1) {
            return false;
        } else {
            Set<Integer> set = new HashSet<>();
            for (int num : nums) {
                if (!set.add(num)) {
                    return true;
                }
            }
            return false;
        }
    }

    /**
     * Given a column title as appear in an Excel sheet, return its corresponding column number.
     * <p>
     * For example:
     * <p>
     * A -> 1
     * B -> 2
     * C -> 3
     * ...
     * Z -> 26
     * AA -> 27
     * AB -> 28
     *
     * @param s
     * @return
     */
    public int titleToNumber(String s) {
        int sum = 0;
        for (int i = s.length() - 1, j = 0; i >= 0; --i, ++j) {
            int value = (int) s.charAt(i) - 64;
            sum += value * Math.pow(26, j);
        }
        return sum;
    }

    /**
     * Given two strings s and t, write a function to determine if t is an anagram of s.
     * <p>
     * For example,
     * s = "anagram", t = "nagaram", return true.
     * s = "rat", t = "car", return false.
     * <p>
     * Note:
     * You may assume the string contains only lowercase alphabets.
     * <p>
     * Follow up:
     * What if the inputs contain unicode characters? How would you adapt your solution to such case?
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        } else if (s.isEmpty()) {
            return true;
        } else {
            Map<String, Integer> map = new HashMap<>();
            for (int i = 0; i < s.length(); i++) {
                String key = String.valueOf(s.charAt(i));
                if (map.containsKey(key)) {
                    map.put(key, map.get(key) + 1);
                } else {
                    map.put(key, 1);
                }
            }
            for (int i = 0; i < t.length(); i++) {
                String key = String.valueOf(t.charAt(i));
                if (map.containsKey(key)) {
                    map.put(key, map.get(key) - 1);
                } else {
                    return false;
                }
            }
            for (Integer val : map.values()) {
                if (val != 0) {
                    return false;
                }
            }
            return true;
        }
    }


    /**
     * Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.
     * <p>
     * According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes v and w as the lowest node in T that has both v and w as descendants (where we allow a node to be a descendant of itself).”
     * <p>
     * _______6______
     * /              \
     * ___2__          ___8__
     * /      \        /      \
     * 0      _4       7       9
     * /  \
     * 3   5
     * For example, the lowest common ancestor (LCA) of nodes 2 and 8 is 6. Another example is LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if ((p.val < root.val && q.val > root.val) || (p.val > root.val && q.val < root.val)) {
            return root;
        } else if (p.val < root.val && q.val < root.val) {
            return lowestCommonAncestor(root.left, p, q);
        } else if (p.val > root.val && q.val > root.val) {
            return lowestCommonAncestor(root.right, p, q);
        } else {
            return p.val == root.val ? p : q;
        }
    }

    /**
     * Write a function that takes an unsigned integer and returns the number of ’1' bits it has (also known as the Hamming weight).
     * <p>
     * For example, the 32-bit integer ’11' has binary representation 00000000000000000000000000001011, so the function should return 3.
     *
     * @param n
     * @return
     */
    public int hammingWeight(long n) {
        int sum = 0;
        while (n != 0) {
            if ((n & 1) == 1) {
                sum++;
            }
            n = n >>> 1;
        }
        return sum;
    }

    /**
     * Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.
     * <p>
     * You may assume that the array is non-empty and the majority element always exist in the array.
     *
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (map.containsKey(num)) {
                map.put(num, map.get(num) + 1);
            } else {
                map.put(num, 1);
            }
        }
        int result = 0;
        int rk = -1;
        for (Integer key : map.keySet()) {
            if (map.get(key) > result) {
                result = map.get(key);
                rk = key;
            }
        }
        return rk;
    }

    public int majorityElementOn(int[] nums) {
        int curEle = -1;
        int count = 0;
        for (int num : nums) {
            if (count != 0) {
                if (num == curEle) {
                    count++;
                } else {
                    count--;
                }
            }
            if (count == 0) {
                curEle = num;
                count = 1;
            }
        }
        return curEle;
    }

    /**
     * Given a roman numeral, convert it to an integer.
     * <p>
     * Input is guaranteed to be within the range from 1 to 3999.
     *
     * @param s
     * @return
     */
    public int romanToInt(String s) {
        Map<String, Integer> map = new HashMap<>();
        map.put("I", 1);
        map.put("V", 5);
        map.put("X", 10);
        map.put("L", 50);
        map.put("C", 100);
        map.put("D", 500);
        map.put("M", 1000);
        if (s.length() == 1) {
            return map.get(s);
        } else {
            int result = map.get(String.valueOf(s.charAt(s.length() - 1)));
            int i = s.length() - 2;
            while (i >= 0) {
                int iv = map.get(String.valueOf(s.charAt(i)));
                int beforeIv = map.get(String.valueOf(s.charAt(i + 1)));
                if (iv >= beforeIv) {
                    result += iv;
                } else {
                    result -= iv;
                }
                --i;
            }
            return result;
        }
    }

    /**
     * Reverse a singly linked list.
     *
     * @param head
     * @return
     */
    public ListNode reverseList(ListNode head) {
        Stack<Integer> stack = new Stack<>();
        pushList(head, stack);
        popList(head, stack);
        return head;
    }

    public void popList(ListNode head, Stack<Integer> stack) {
        if (!stack.empty()) {
            head.val = stack.pop();
            popList(head.next, stack);
        }
    }

    public void pushList(ListNode node, Stack<Integer> stack) {
        if (node != null) {
            stack.push(node.val);
            pushList(node.next, stack);
        }
    }


    /**
     * Given a sorted linked list, delete all duplicates such that each element appear only once.
     * <p>
     * For example,
     * Given 1->1->2, return 1->2.
     * Given 1->1->2->3->3, return 1->2->3.
     *
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) {
            return head;
        }
        ListNode next = head.next;
        ListNode curNode = head;
        while (next != null) {
            if (curNode.val == next.val) {
                next = next.next;
            } else {
                curNode.next = next;
                curNode = next;
            }
        }
        curNode.next = null;
        return head;
    }

    /**
     * Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.
     * <p>
     * For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].
     * <p>
     * Note:
     * You must do this in-place without making a copy of the array.
     * Minimize the total number of operations.
     *
     * @param nums
     */
    public void moveZeroes(int[] nums) {
        if (nums.length >= 2) {
            int i = 0;
            while (i < nums.length) {
                if (nums[i] == 0) {
                    int j = i + 1;
                    if (j < nums.length) {
                        while (j < nums.length - 1 && nums[j] == 0) {
                            ++j;
                        }
                        if (nums[i] != nums[j]) {
                            swap(nums, i, j);
                        }
                    }
                }
                ++i;
            }
        }
    }

    public void moveZeroesInOn(int[] nums) {
        if (nums.length >= 2) {
            int zeroCount = 0;
            for (int i = 0; i < nums.length; i++) {
                if (nums[i] == 0) {
                    zeroCount++;
                } else {
                    nums[i - zeroCount] = nums[i];
                }
            }
            for (int i = 0; i < zeroCount; i++) {
                nums[nums.length - i - 1] = 0;
            }
        }
    }

    public void swap(int[] nums, int i, int j) {
        nums[i] = nums[i] + nums[j];
        nums[j] = nums[i] - nums[j];
        nums[i] = nums[i] - nums[j];
    }

    /**
     * You are climbing a stair case. It takes n steps to reach to the top.
     * <p>
     * Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
     *
     * @param n
     * @return
     */
    public int climbStairs(int n) {
        if (n == 1 || n == 0)
            return n;
        int count1 = 1;
        int count2 = 1;


        for (int i = 2; i <= n; i++) {
            int temp = count2;
            count2 = temp + count1;
            count1 = temp;
        }
        return count2;
    }

    /**
     * Write a program to check whether a given number is an ugly number.
     * <p>
     * Ugly numbers are positive numbers whose prime factors only include 2, 3, 5. For example, 6, 8 are ugly while 14 is not ugly since it includes another prime factor 7.
     * <p>
     * Note that 1 is typically treated as an ugly number.
     *
     * @param num
     * @return
     */
    public boolean isUgly(int num) {
        if (num > 0) {
            for (int i = 2; i < 6; i++) {
                while (num % i == 0) {
                    num /= i;
                }
            }
        }
        return num == 1;
    }

    /**
     * Write an algorithm to determine if a number is "happy".
     * <p>
     * A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.
     * <p>
     * Example: 19 is a happy number
     * <p>
     * 12 + 92 = 82
     * 82 + 22 = 68
     * 62 + 82 = 100
     * 12 + 02 + 02 = 1
     *
     * @param n
     * @return
     */
    public boolean isHappy(int n) {
        if (n == 1) {
            return true;
        }
        Set<Integer> history = new HashSet<>();
        history.add(n);
        while (n != 1) {
            int temp = n;
            int sum = 0;
            while (temp != 0) {
                sum += Math.pow((temp % 10), 2);
                temp /= 10;
            }
            if (history.contains(sum)) {
                return false;
            } else {
                n = sum;
                history.add(n);
            }
        }
        return n == 1;
    }

    /**
     * Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }

    /**
     * Given an integer, write a function to determine if it is a power of two.
     *
     * @param n
     * @return
     */
    public boolean isPowerOfTwo(int n) {
        if (n == 1) {
            return true;
        }
        if (n == 0) {
            return false;
        }
        int temp = n >> 1;
        if (temp * 2 == n) {
            return isPowerOfTwo(temp);
        } else {
            return false;
        }
    }

    /**
     * Given a binary tree, determine if it is height-balanced.
     * <p>
     * For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.
     *
     * @param root
     * @return
     */
    public boolean isBalanced(TreeNode root) {
        return dfsHeight(root) != -1;
    }

    private int dfsHeight(TreeNode root) {
        if (root == null) return -1;
        int leftHeight = dfsHeight(root.left);
        if (leftHeight == -1) return -1;
        int rightHeight = dfsHeight(root.right);
        if (rightHeight == -1) return -1;
        if (Math.abs(leftHeight - rightHeight) > 1) {
            return -1;
        } else {
            return Math.max(leftHeight, rightHeight) + 1;
        }
    }

    /**
     * Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
     * <p>
     * For example, this binary tree is symmetric:
     * <p>
     * 1
     * / \
     * 2   2
     * / \ / \
     * 3  4 4  3
     * But the following is not:
     * 1
     * / \
     * 2   2
     * \   \
     * 3    3
     * Note:
     * Bonus points if you could solve it both recursively and iteratively.
     *
     * @param root
     * @return
     */
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }

        return isSymmetricHelp(root.left, root.right);
    }

    private boolean isSymmetricHelp(TreeNode left, TreeNode right) {
        if (left == null || right == null) {
            return left == right;
        }
        if (left.val != right.val) {
            return false;
        }
        return isSymmetricHelp(left.left, right.right) && isSymmetricHelp(left.right, right.left);
    }

    /**
     * Given an array and a value, remove all instances of that value in place and return the new length.
     * <p>
     * The order of elements can be changed. It doesn't matter what you leave beyond the new length.
     *
     * @param nums
     * @param val
     * @return
     */
    public int removeElement(int[] nums, int val) {
        int size = nums.length;
        if (size == 1) {
            return nums[0] == val ? 0 : 1;
        }
        if (size == 0) {
            return 0;
        }
        int i = 0, j = size - 1, length = 0;
        while (true) {
            while (i < size && nums[i] != val) {
                i++;
                length++;
            }
            while (j >= 0 && nums[j] == val) {
                j--;
            }
            if (i > j) {
                break;
            } else {
                int temp = nums[j];
                nums[j] = nums[i];
                nums[i] = temp;
            }
        }
        return length;
    }

    /**
     * Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to root).
     * <p>
     * For example:
     * Given binary tree {3,9,20,#,#,15,7},
     * 3
     * / \
     * 9  20
     * /  \
     * 15   7
     * return its bottom-up level order traversal as:
     * [
     * [15,7],
     * [9,20],
     * [3]
     * ]
     *
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> lists = new LinkedList<>();
        if (root != null) {
            List<TreeNode> fronties = new ArrayList<>();
            fronties.add(root);
            while (fronties.size() > 0) {
                List<Integer> ele = new LinkedList<>();
                List<TreeNode> temp = new ArrayList<>();
                for (TreeNode fronty : fronties) {
                    ele.add(fronty.val);
                    if (fronty.left != null) {
                        temp.add(fronty.left);
                    }
                    if (fronty.right != null) {
                        temp.add(fronty.right);
                    }
                }
                lists.add(0, ele);
                fronties = temp;
            }
        }
        return lists;
    }

    /**
     * Given a sorted array, remove the duplicates in place such that each element appear only once and return the new length.
     * <p>
     * Do not allocate extra space for another array, you must do this in place with constant memory.
     * <p>
     * For example,
     * Given input array nums = [1,1,2],
     * <p>
     * Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively. It doesn't matter what you leave beyond the new length.
     *
     * @param nums
     * @return
     */
    public int removeDuplicates(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return 1;
        int i = 0, j = 1;
        int size = nums.length;
        while (i < nums.length && j < nums.length) {
            if (nums[i] == nums[j]) {
                j++;
                size--;
            } else {
                int temp = nums[i + 1];
                nums[i + 1] = nums[j];
                nums[j] = temp;
                i++;
                j++;
            }
        }
        return size;
    }

    /**
     * You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
     * <p>
     * Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.
     *
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        int n = nums.length, pre = 0, cur = 0;
        for (int i = 0; i < n; i++) {
            int temp = Math.max(pre + nums[i], cur);
            pre = cur;
            cur = temp;
        }
        return cur;
    }

    /**
     * Given a non-negative number represented as an array of digits, plus one to the number.
     * <p>
     * The digits are stored such that the most significant digit is at the head of the list.
     *
     * @param digits
     * @return
     */
    public int[] plusOne(int[] digits) {
        int temp = 1;
        int top = 0;
        for (int i = digits.length - 1; i >= 0; i--) {
            int plus = digits[i] + temp;
            if (plus < 10) {
                temp = 0;
            }
            if (plus >= 10 && i == 0) {
                top = 1;
            }
            digits[i] = plus % 10;
        }
        if (top == 1) {
            int[] result = new int[digits.length + 1];
            result[0] = 1;
            for (int i = 1; i < result.length; i++) {
                result[i] = digits[i - 1];
            }
            return result;
        }

        return digits;
    }

    /**
     * Given numRows, generate the first numRows of Pascal's triangle.
     * <p>
     * For example, given numRows = 5,
     * Return
     * <p>
     * [
     * [1],
     * [1,1],
     * [1,2,1],
     * [1,3,3,1],
     * [1,4,6,4,1]
     * ]
     *
     * @param numRows
     * @return
     */
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> result = new LinkedList<>();
        for (int i = 0; i < numRows; i++) {
            List<Integer> temp = new ArrayList<>();
            if (result.size() == 0) {
                temp.add(1);
                result.add(temp);
            } else {
                List<Integer> content = result.get(result.size() - 1);
                if (content.size() == 1) {
                    temp.add(1);
                    temp.add(1);
                    result.add(temp);
                } else {
                    temp.add(1);
                    int m = 0, n = 1;
                    while (n < content.size()) {
                        temp.add(content.get(m) + content.get(n));
                        m++;
                        n++;
                    }
                    temp.add(1);
                    result.add(temp);
                }
            }
        }
        return result;
    }

    /**
     * Given an integer n, return the number of trailing zeroes in n!.
     * <p>
     * Note: Your solution should be in logarithmic time complexity.
     *
     * @param n
     * @return
     */
    public int trailingZeroes(int n) {
        return n == 0 ? 0 : n / 5 + trailingZeroes(n / 5);
    }

    /**
     * Given an index k, return the kth row of the Pascal's triangle.
     * <p>
     * For example, given k = 3,
     * Return [1,3,3,1].
     *
     * @param rowIndex
     * @return
     */
    public List<Integer> getRow(int rowIndex) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i <= rowIndex; i++) {
            if (i == 0) {
                result.add(1);
            } else if (i == 1) {
                result.add(1);
            } else {
                int size = result.size();
                int pre = result.get(0);
                for (int j = 1; j < size; j++) {
                    int temp = result.get(j);
                    result.set(j, result.get(j) + pre);
                    pre = temp;
                }
                result.add(1);
            }
        }
        return result;
    }

    /**
     * Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.
     * <p>
     * For example:
     * Given the below binary tree and sum = 22,
     * 5
     * / \
     * 4   8
     * /   / \
     * 11  13  4
     * /  \      \
     * 7    2      1
     * return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22.
     *
     * @param root
     * @param sum
     * @return
     */
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) return false;
        if (root.left == null && root.right == null) return root.val == sum;
        int temp = sum - root.val;
        return hasPathSum(root.left, temp) || hasPathSum(root.right, temp);
    }

    /**
     * Determine whether an integer is a palindrome. Do this without extra space.
     *
     * @param x
     * @return
     */
    public boolean isPalindrome(int x) {
        int sum = 0, target = x;
        while (x > 0) {
            int temp = x % 10;
            x /= 10;
            sum = sum * 10 + temp;
        }
        return sum == target;
    }

    /**
     * Write a program to find the node at which the intersection of two singly linked lists begins.
     * <p>
     * <p>
     * For example, the following two linked lists:
     * <p>
     * A:          a1 → a2
     * ↘
     * c1 → c2 → c3
     * ↗
     * B:     b1 → b2 → b3
     * begin to intersect at node c1.
     * <p>
     * <p>
     * Notes:
     * <p>
     * If the two linked lists have no intersection at all, return null.
     * The linked lists must retain their original structure after the function returns.
     * You may assume there are no cycles anywhere in the entire linked structure.
     * Your code should preferably run in O(n) time and use only O(1) memory.
     *
     * @param headA
     * @param headB
     * @return
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null && headB == null) return null;
        ListNode a = headA;
        ListNode b = headB;
        while (a != b) {
            a = a != null ? a.next : headB;
            b = b != null ? b.next : headA;
        }
        return a;
    }


    /**
     * Given a binary tree, find its minimum depth.
     * <p>
     * The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
     *
     * @param root
     * @return
     */
    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) return 1;
        int depth = 1;
        List<TreeNode> frontiers = new ArrayList<>();
        frontiers.add(root);
        while (frontiers.size() > 0) {
            List<TreeNode> temp = new ArrayList<>();
            for (TreeNode frontier : frontiers) {
                if (frontier.left != null) temp.add(frontier.left);
                if (frontier.right != null) temp.add(frontier.right);
                if (frontier.left == null && frontier.right == null) return depth;
            }
            frontiers = temp;
            depth++;
        }
        return depth;
    }

    /**
     * Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.
     * <p>
     * Note:
     * You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2. The number of elements initialized in nums1 and nums2 are m and n respectively.
     *
     * @param nums1
     * @param m
     * @param nums2
     * @param n
     */
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1;
        int j = n - 1;
        int k = m + n - 1;
        while (i >= 0 && j >= 0) {
            if (nums1[i] > nums2[j])
                nums1[k--] = nums1[i--];
            else
                nums1[k--] = nums2[j--];
        }
        while (j >= 0)
            nums1[k--] = nums2[j--];
    }

    /**
     * Reverse bits of a given 32 bits unsigned integer.
     * <p>
     * For example, given input 43261596 (represented in binary as 00000010100101000001111010011100), return 964176192 (represented in binary as 00111001011110000010100101000000).
     * <p>
     * Follow up:
     * If this function is called many times, how would you optimize it?
     *
     * @param n
     * @return
     */
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        int reverse = 0;
        for (int i = 0; i < 32; i++) {
            reverse += n & 1;
            if (i < 31) {
                reverse = reverse << 1;
            }
            n >>>= 1;
        }
        return reverse;
    }

    /**
     * Determine if a Sudoku is valid, according to: Sudoku Puzzles - The Rules.
     * <p>
     * The Sudoku board could be partially filled, where empty cells are filled with the character '.'.
     * <p>
     * <p>
     * A partially filled sudoku which is valid.
     * <p>
     * Note:
     * A valid Sudoku board (partially filled) is not necessarily solvable. Only the filled cells need to be validated.
     *
     * @param board
     * @return
     */
    public boolean isValidSudoku(char[][] board) {
        for (int i = 0; i < 9; i++) {
            if (!isParticallyValid(board, i, 0, i, 8)) return false;
            if (!isParticallyValid(board, 0, i, 8, i)) return false;
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (!isParticallyValid(board, i * 3, j * 3, i * 3 + 2, j * 3 + 2)) return false;
            }
        }
        return true;
    }

    private boolean isParticallyValid(char[][] board, int x1, int y1, int x2, int y2) {
        Set singleSet = new HashSet();
        for (int i = x1; i <= x2; i++) {
            for (int j = y1; j <= y2; j++) {
                if (board[i][j] != '.') if (!singleSet.add(board[i][j])) return false;
            }
        }
        return true;
    }

    /**
     * Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string.
     * <p>
     * If the last word does not exist, return 0.
     * <p>
     * Note: A word is defined as a character sequence consists of non-space characters only.
     * <p>
     * For example,
     * Given s = "Hello World",
     * return 5.
     *
     * @param s
     * @return
     */
    public int lengthOfLastWord(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        } else {
            int sum = 0;
            for (int i = s.length() - 1; i >= 0; i--) {
                if (!String.valueOf(s.charAt(i)).equals(" ")) {
                    sum++;
                } else {
                    if (sum > 0) {
                        break;
                    }
                    continue;
                }
            }
            return sum;
        }
    }

    int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        int x = Math.min(G, C) > Math.max(E, A) ? (Math.min(G, C) - Math.max(E, A)) : 0;
        int y = Math.min(D, H) > Math.max(B, F) ? (Math.min(D, H) - Math.max(B, F)) : 0;
        return (D - B) * (C - A) + (G - E) * (H - F) - x * y;
    }

    /**
     * Compare two version numbers version1 and version2.
     * If version1 > version2 return 1, if version1 < version2 return -1, otherwise return 0.
     * <p>
     * You may assume that the version strings are non-empty and contain only digits and the . character.
     * The . character does not represent a decimal point and is used to separate number sequences.
     * For instance, 2.5 is not "two and a half" or "half way to version three", it is the fifth second-level revision of the second first-level revision.
     * <p>
     * Here is an example of version numbers ordering:
     * <p>
     * 0.1 < 1.1 < 1.2 < 13.37
     *
     * @param version1
     * @param version2
     * @return
     */
    public int compareVersion(String version1, String version2) {
        String[] ver1s = version1.split("\\.");
        String[] ver2s = version2.split("\\.");
        int i = 0, j = 0, ver = 0;
        while (i < ver1s.length && j < ver2s.length) {
            Integer pv1 = Integer.valueOf(ver1s[i]);
            Integer pv2 = Integer.valueOf(ver2s[j]);
            if (pv1 > pv2) {
                ver = 1;
                break;
            } else if (pv1 < pv2) {
                ver = -1;
                break;
            } else {
                i++;
                j++;
                continue;
            }
        }
        if (ver == 0) {
            while (i < ver1s.length) {
                if (Integer.valueOf(ver1s[i]) > 0) {
                    ver = 1;
                    return ver;
                }
                i++;
            }
            while (j < ver2s.length) {
                if (Integer.valueOf(ver2s[j]) > 0) {
                    ver = -1;
                    return ver;
                }
                j++;
            }
        }
        return ver;
    }

    /**
     * Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and the difference between i and j is at most k.
     *
     * @param nums
     * @param k
     * @return
     */
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            if (i > k) set.remove(nums[i - k - 1]);
            if (!set.add(nums[i])) return true;
        }
        return false;
    }

    /**
     * Given a linked list, remove the nth node from the end of list and return its head.
     * <p>
     * For example,
     * <p>
     * Given linked list: 1->2->3->4->5, and n = 2.
     * <p>
     * After removing the second node from the end, the linked list becomes 1->2->3->5.
     * Note:
     * Given n will always be valid.
     * Try to do this in one pass.
     *
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode start = new ListNode(0);
        ListNode slow = start, fast = start;
        slow.next = head;

        //Move fast in front so that the gap between slow and fast becomes n
        for (int i = 1; i <= n + 1; i++) {
            fast = fast.next;
        }
        //Move fast to the end, maintaining the gap
        while (fast != null) {
            slow = slow.next;
            fast = fast.next;
        }
        //Skip the desired node
        slow.next = slow.next.next;
        return start.next;
    }

    /**
     * Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
     * <p>
     * The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.
     *
     * @param s
     * @return
     */
    public boolean isValid(String s) {
        Stack<String> charsStack = new Stack<>();
        Map<String, String> kv = new HashMap<>();
        kv.put("}", "{");
        kv.put(")", "(");
        kv.put("]", "[");
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == '[' || ch == '(' || ch == '{') {
                charsStack.push(String.valueOf(ch));
            }
            if (ch == ']' || ch == ')' || ch == '}') {
                if (charsStack.size() == 0) {
                    return false;
                }
                if (charsStack.size() > 0 && !kv.get(String.valueOf(ch)).equals(charsStack.pop())) {
                    return false;
                }
            }
        }
        if (charsStack.size() > 0) return false;
        return true;
    }

    /**
     * Given two strings s and t, determine if they are isomorphic.
     * <p>
     * Two strings are isomorphic if the characters in s can be replaced to get t.
     * <p>
     * All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character but a character may map to itself.
     * <p>
     * For example,
     * Given "egg", "add", return true.
     * <p>
     * Given "foo", "bar", return false.
     * <p>
     * Given "paper", "title", return true.
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isIsomorphic(String s, String t) {
        if (s == null || t == null) return false;
        if (s.length() != t.length()) return false;
        Map<String, String> kv = new HashMap<>();
        Set<String> vs = new HashSet<>();
        for (int i = 0; i < s.length(); i++) {
            String k = String.valueOf(s.charAt(i));
            if (!kv.containsKey(k)) {
                String value = String.valueOf(t.charAt(i));
                if (vs.contains(value)) return false;
                kv.put(k, value);
                vs.add(value);
            } else if (!kv.get(k).equals(String.valueOf(t.charAt(i)))) {
                return false;
            }
        }
        return true;
    }

    /**
     * Given a pattern and a string str, find if str follows the same pattern.
     * <p>
     * Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in str.
     * <p>
     * Examples:
     * pattern = "abba", str = "dog cat cat dog" should return true.
     * pattern = "abba", str = "dog cat cat fish" should return false.
     * pattern = "aaaa", str = "dog cat cat dog" should return false.
     * pattern = "abba", str = "dog dog dog dog" should return false.
     * Notes:
     * You may assume pattern contains only lowercase letters, and str contains lowercase letters separated by a single space.
     *
     * @param pattern
     * @param str
     * @return
     */
    public boolean wordPattern(String pattern, String str) {
        if (pattern == null || str == null) return false;
        if (pattern.length() != str.split(" ").length) return false;
        String[] ss = str.split(" ");
        Map<String, String> kv = new HashMap<>();
        Set<String> vs = new HashSet<>();
        for (int i = 0; i < pattern.length(); i++) {
            String k = String.valueOf(pattern.charAt(i));
            String value = ss[i];
            if (!kv.containsKey(k)) {
                if (vs.contains(value)) return false;
                kv.put(k, value);
                vs.add(value);
            } else if (!kv.get(k).equals(value)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Remove all elements from a linked list of integers that have value val.
     * <p>
     * Example
     * Given: 1 --> 2 --> 6 --> 3 --> 4 --> 5 --> 6, val = 6
     * Return: 1 --> 2 --> 3 --> 4 --> 5
     *
     * @param head
     * @param val
     * @return
     */
    public ListNode removeElements(ListNode head, int val) {
        while (head != null && head.val == val) {
            head = head.next;
        }
        if (head == null) return head;
        ListNode n1 = head;
        while (n1 != null) {
            if (n1.next != null && n1.next.val == val) {
                n1.next = n1.next.next;
            } else {
                n1 = n1.next;
            }
        }
        return head;
    }

    /**
     * The count-and-say sequence is the sequence of integers beginning as follows:
     * 1, 11, 21, 1211, 111221, ...
     * <p>
     * 1 is read off as "one 1" or 11.
     * 11 is read off as "two 1s" or 21.
     * 21 is read off as "one 2, then one 1" or 1211.
     * Given an integer n, generate the nth sequence.
     * <p>
     * Note: The sequence of integers will be represented as a string.
     *
     * @param n
     * @return
     */
    public String countAndSay(int n) {
        if (n <= 0) return null;
        String s = "1";
        for (int i = 1; i < n; i++) {
            s = genSeq(s);
        }
        return s;
    }

    public String genSeq(String pre) {
        String ele = null;
        String result = "";
        Integer count = 0;
        for (int i = 0; i < pre.length(); i++) {
            String cur = String.valueOf(pre.charAt(i));
            if (ele != null && ele.equals(cur)) {
                count++;
            } else {
                if (ele != null) {
                    result += count + ele;
                }
                ele = cur;
                count = 1;
            }
        }
        result += count + ele;
        System.out.println(result);
        return result;
    }

    /**
     * You are playing the following Bulls and Cows game with your friend: You write down a number and ask your friend to guess what the number is. Each time your friend makes a guess, you provide a hint that indicates how many digits in said guess match your secret number exactly in both digit and position (called "bulls") and how many digits match the secret number but locate in the wrong position (called "cows"). Your friend will use successive guesses and hints to eventually derive the secret number.
     * <p>
     * For example:
     * <p>
     * Secret number:  "1807"
     * Friend's guess: "7810"
     * Hint: 1 bull and 3 cows. (The bull is 8, the cows are 0, 1 and 7.)
     * Write a function to return a hint according to the secret number and friend's guess, use A to indicate the bulls and B to indicate the cows. In the above example, your function should return "1A3B".
     * <p>
     * Please note that both secret number and friend's guess may contain duplicate digits, for example:
     * <p>
     * Secret number:  "1123"
     * Friend's guess: "0111"
     * In this case, the 1st 1 in friend's guess is a bull, the 2nd or 3rd 1 is a cow, and your function should return "1A1B".
     * You may assume that the secret number and your friend's guess only contain digits, and their lengths are always equal.
     *
     * @param secret
     * @param guess
     * @return
     */
    public String getHint(String secret, String guess) {
        if (secret == null || guess == null) return null;
        int bulls = 0, cows = 0;
        int[] numbers = new int[10];
        for (int i = 0; i < secret.length(); i++) {
            int s = Character.getNumericValue(secret.charAt(i));
            int g = Character.getNumericValue(guess.charAt(i));
            if (s == g) {
                bulls++;
            } else {
                if (numbers[s] < 0) cows++;
                if (numbers[g] > 0) cows++;
                numbers[s]++;
                numbers[g]--;
            }
        }
        return bulls + "A" + cows + "B";
    }

    /**
     * Write a function to find the longest common prefix string amongst an array of strings.
     *
     * @param strs
     * @return
     */
    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) return "";
        if (strs.length == 1) return strs[0];
        String common = null;
        for (int i = 0; i < strs.length; i++) {
            common = getLongestPrefixFor2(common, strs[i]);
            if ("".equals(common)) {
                break;
            }
        }
        return common;
    }

    public String getLongestPrefixFor2(String str1, String str2) {
        if (str1 == null) return str2;
        int length = str1.length() < str2.length() ? str1.length() : str2.length();
        int commonLen = 0;
        for (int i = 0; i < length; i++) {
            if (str1.charAt(i) == str2.charAt(i)) {
                commonLen++;
            } else {
                break;
            }
        }
        return commonLen > 0 ? str1.substring(0, commonLen) : "";
    }

    /**
     * Given two binary strings, return their sum (also a binary string).
     * <p>
     * For example,
     * a = "11"
     * b = "1"
     * Return "100".
     *
     * @param a
     * @param b
     * @return
     */
    public String addBinary(String a, String b) {
        if (a == null || b == null) return null;
        if (a.length() == 0 || b.length() == 0) return null;
        int delta = a.length() > b.length() ? a.length() - b.length() : b.length() - a.length();
        if (a.length() > b.length()) {
            for (int i = 0; i < delta; i++) {
                b = "0" + b;
            }
        } else {
            for (int i = 0; i < delta; i++) {
                a = "0" + a;
            }
        }
        int digit = 0;
        String result = "";
        for (int i = a.length() - 1; i >= 0; i--) {
            int av = Character.getNumericValue(a.charAt(i));
            int bv = Character.getNumericValue(b.charAt(i));
            int sum = av + bv + digit;
            if (sum > 1) {
                result = sum % 2 + result;
                digit = 1;
            } else {
                result = sum + result;
                digit = 0;
            }
        }
        if (digit > 0) {
            result = "1" + result;
        }
        return result;
    }

    /**
     * Given a singly linked list, determine if it is a palindrome.
     * <p>
     * Follow up:
     * Could you do it in O(n) time and O(1) space?
     *
     * @param head
     * @return
     */
    public boolean isPalindrome(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        if (fast != null) slow = slow.next;// if the number of nodes is odd
        slow = reverseListInConstant(slow);
        while (slow != null && slow.val == head.val) {
            slow = slow.next;
            head = head.next;
        }
        return slow == null;
    }

    private ListNode reverseListInConstant(ListNode head) {
        ListNode pre = null;
        while (head != null) {
            ListNode next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }

    /**
     * Given a binary tree, return all root-to-leaf paths.
     * <p>
     * For example, given the following binary tree:
     * <p>
     * 1
     * /   \
     * 2     3
     * \
     * 5
     * All root-to-leaf paths are:
     * <p>
     * ["1->2->5", "1->3"]
     *
     * @param root
     * @return
     */
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> paths = new ArrayList<>();
        traverse(null, root, paths);
        return paths;
    }

    public void traverse(String prePath, TreeNode root, List<String> paths) {
        if (root != null) {
            String path = prePath != null ? prePath + "->" + root.val : String.valueOf(root.val);
            if (root.left == null && root.right == null) {
                paths.add(path);
            }
            if (root.left != null) {
                traverse(path, root.left, paths);
            }
            if (root.right != null) {
                traverse(path, root.right, paths);
            }
        }
    }

    /**
     * Implement strStr().
     * <p>
     * Returns the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.
     *
     * @param haystack
     * @param needle
     * @return
     */
    public int strStr(String haystack, String needle) {
        if (haystack == null || needle == null) return -1;
        if (needle.length() > haystack.length()) return -1;
        if (haystack.length() == 0 || needle.length() == 0) return 0;
        for (int i = 0; i <= haystack.length() - needle.length(); i++) {
            for (int j = 0; j < needle.length(); j++) {
                if (haystack.charAt(i + j) != needle.charAt(j)) break;
                if (j == needle.length() - 1) return i;
            }
        }
        return -1;
    }

    /**
     * Reverse digits of an integer.
     * <p>
     * Example1: x = 123, return 321
     * Example2: x = -123, return -321
     *
     * @param x
     * @return
     */
    public int reverse(int x) {
        if (x > Integer.MAX_VALUE) return 0;
        int count = 0;
        int reverse = 0;
        Stack<Integer> xs = new Stack<>();
        while (x != 0) {
            xs.push(x % 10);
            x /= 10;
            count++;
        }
        for (int i = 0; i < count; i++) {
            reverse += (xs.pop() * Math.pow(10, i));
            if (reverse == Integer.MAX_VALUE || reverse == Integer.MIN_VALUE) return 0;
        }
        return x < 0 ? -reverse : reverse;
    }

    /**
     * Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
     * <p>
     * For example,
     * "A man, a plan, a canal: Panama" is a palindrome.
     * "race a car" is not a palindrome.
     * <p>
     * Note:
     * Have you consider that the string might be empty? This is a good question to ask during an interview.
     * <p>
     * For the purpose of this problem, we define empty string as valid palindrome.
     *
     * @param s
     * @return
     */
    public boolean isPalindrome(String s) {
        int i = 0, j = s.length() - 1;
        while (i < j) {
            while (i < j && !Character.isLetterOrDigit(s.charAt(i))) {
                ++i;
            }
            while (i < j && !Character.isLetterOrDigit(s.charAt(j))) {
                --j;
            }
            if (Character.toLowerCase(s.charAt(i)) != Character.toLowerCase(s.charAt(j))) return false;
            ++i;
            --j;
        }
        return true;
    }

    /**
     * The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)
     * <p>
     * P   A   H   N
     * A P L S I I G
     * Y   I   R
     * And then read line by line: "PAHNAPLSIIGYIR"
     * Write the code that will take a string and make this conversion given a number of rows:
     * <p>
     * string convert(string text, int nRows);
     * convert("PAYPALISHIRING", 3) should return "PAHNAPLSIIGYIR".
     *
     * @param s
     * @param numRows
     * @return
     */
    public String convert(String s, int numRows) {
        if (numRows > s.length() || numRows == 1) return s;
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < numRows; i++) {
            sb.append(s.charAt(i));
            int j = i;
            while (j < s.length()) {
                int index1 = j + (numRows - i) * 2 - 2;
                if (index1 > j && index1 < s.length()) {
                    sb.append(s.charAt(index1));
                }
                if ((index1 + i * 2) > index1) {
                    index1 = index1 + i * 2;
                    if (index1 < s.length()) {
                        sb.append(s.charAt(index1));
                    }
                }
                j = index1;
            }
        }
        return sb.toString();
    }

    /**
     * Description:
     * <p>
     * Count the number of prime numbers less than a non-negative number, n.
     *
     * @param n
     * @return
     */
    public int countPrimes(int n) {
        if (n > 1) {
            boolean[] primeRec = new boolean[n];
            int count = 0;
            for (int i = 0; i < n; i++) {
                primeRec[i] = true;
            }
            primeRec[0] = false;
            primeRec[1] = false;
            for (int i = 4; i < n; i++) {
                if (i % 2 == 0) primeRec[i] = false;
            }
            for (int i = 3; i < Math.sqrt(n); i += 2) {
                for (int j = i; j * i < n; j += 2) {
                    primeRec[j * i] = false;
                }
            }
            for (int i = 0; i < n; i++) {
                if (primeRec[i]) count++;
            }
            return count;
        } else {
            return 0;
        }
    }

    /**
     * Implement atoi to convert a string to an integer.
     * <p>
     * Hint: Carefully consider all possible input cases. If you want a challenge, please do not see below and ask yourself what are the possible input cases.
     * <p>
     * Notes: It is intended for this problem to be specified vaguely (ie, no given input specs). You are responsible to gather all the input requirements up front.
     * <p>
     * Update (2015-02-10):
     * The signature of the C++ function had been updated. If you still see your function signature accepts a const char * argument, please click the reload button  to reset your code definition.
     *
     * @param str
     * @return
     */
    public int myAtoi(String str) {
        str = str.trim();
        if (str.length() > 0) {
            boolean positive = str.charAt(0) != '-' ? true : false;
            Stack<Integer> stack = new Stack<>();
            int firstVal = Character.getNumericValue(str.charAt(0));
            if (firstVal > 0 && firstVal < 10) stack.push(firstVal);
            if (firstVal >= 10 && firstVal <= 35) return 0;
            for (int i = 1; i < str.length(); i++) {
                int value = Character.getNumericValue(str.charAt(i));
                if (value != -1 && value != -2 && value < 10) {
                    stack.push(value);
                } else {
                    break;
                }
            }
            int result = 0, power = 0;
            while (!stack.empty()) {
                int val = stack.pop();
                if (positive) {
                    result += Math.pow(10, power) * val;
                } else {
                    result -= Math.pow(10, power) * val;
                }
                power++;
            }
            return result;
        } else {
            return 0;
        }
    }

    /**
     * Given a positive integer, return its corresponding column title as appear in an Excel sheet.
     * <p>
     * For example:
     * <p>
     * 1 -> A
     * 2 -> B
     * 3 -> C
     * ...
     * 26 -> Z
     * 27 -> AA
     * 28 -> AB
     *
     * @param n
     * @return
     */
    public String convertToTitle(int n) {
        Stack<String> str = new Stack<>();
        int last = -1;
        while (n > 26) {
            last = last == 0 ? n % 26 - 1 : n % 26;
            str.push(last == 0 ? "Z" : String.valueOf((char) (last + 64)));
            n /= 26;
        }
        if (n > 0 && n <= 26) {
            if (n != 1 || last != 0) {
                str.push(String.valueOf(last == 0 ? (char) (n + 63) : (char) (n + 64)));
            }
        }
        StringBuilder stringBuilder = new StringBuilder();
        while (!str.empty()) {
            stringBuilder.append(str.pop());
        }
        return stringBuilder.toString();
    }

    /**
     * Rotate an array of n elements to the right by k steps.
     * <p>
     * For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].
     * <p>
     * Note:
     * Try to come up as many solutions as you can, there are at least 3 different ways to solve this problem.
     *
     * @param nums
     * @param k
     */
    public void rotate(int[] nums, int k) {
        int n = nums.length;
        int[] newList = new int[n];
        if (n > 1) {
            k = k % n;
            System.arraycopy(nums, n - k, newList, 0, k);
            System.arraycopy(nums, 0, newList, k, n - k);
            System.arraycopy(newList, 0, nums, 0, n);
        }
    }

    public void rotateWithoutExtraSpace(int[] nums, int k) {

        if(nums == null || nums.length < 2){
            return;
        }

        k = k % nums.length;
        reverse(nums, 0, nums.length - k - 1);
        reverse(nums, nums.length - k, nums.length - 1);
        reverse(nums, 0, nums.length - 1);

    }

    private void reverse(int[] nums, int i, int j){
        int tmp = 0;
        while(i < j){
            tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
            i++;
            j--;
        }
    }

    public static class ValidWordAbbr {

        Map<String,Boolean> map=new HashMap<>();
        Set<String> set=new HashSet<>();

        public ValidWordAbbr(String[] dictionary) {
            for(String word: dictionary){
                if(set.add(word)){
                    int length=word.length();
                    if(length>2){
                        String abbr=String.valueOf(word.charAt(0))+(length-2)+String.valueOf(word.charAt(length-1));
                        map.put(abbr,map.get(abbr)==null?true:false);
                    }else{
                        map.put(word,map.get(word)==null?true:false);
                    }
                }
            }
        }

        public boolean isUnique(String word) {
            int length=word.length();
            String abbr=length>2?(String.valueOf(word.charAt(0))+(length-2)+String.valueOf(word.charAt(length-1))):word;
            if((set.contains(word)&&map.get(abbr))||map.get(abbr)==null){
                return true;
            }else{
                return false;
            }
        }
    }

    public class NumArray {

        private int[] arr;

        public NumArray(int[] nums) {
            arr=new int[nums.length];
            for(int i=0;i<nums.length;i++){
                if(i==0){
                    arr[i]=nums[i];
                }else{
                    arr[i]=arr[i-1]+nums[i];
                }
            }
        }

        public int sumRange(int i, int j) {
            return arr[j]-arr[i];
        }
    }


    public static void main(String[] args) {
        ValidWordAbbr validWordAbbr=new ValidWordAbbr(new String[]{"dog"});
        validWordAbbr.isUnique("dig");
    }
}
