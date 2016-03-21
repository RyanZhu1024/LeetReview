package leet;


import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

/**
 * Created by RyanZhu on 1/20/16.
 */
public class Hard {
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

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

    public String serialize(TreeNode root) {
        if (root == null) {
            return null;
        }
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        StringBuilder sb = new StringBuilder();
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node == null) {
                sb.append("null");
            } else {
                sb.append(node.val);
                queue.add(node.left);
                queue.add(node.right);
            }
            sb.append(",");
        }
        return sb.substring(0, sb.length() - 1);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data == null) return null;
        String[] nodes = data.split(",");
        TreeNode root = new TreeNode(Integer.valueOf(nodes[0]));
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int i = 1;
        while (i < nodes.length) {
            TreeNode node = queue.poll();
            String left = nodes[i];
            String right = nodes[i + 1];
            if (!left.equals("null")) {
                TreeNode lf = new TreeNode(Integer.valueOf(left));
                node.left = lf;
                queue.add(lf);
            } else {
                node.left = null;
            }
            if (!right.equals("null")) {
                TreeNode rt = new TreeNode(Integer.valueOf(right));
                node.right = rt;
                queue.add(rt);
            } else {
                node.right = null;
            }
            i += 2;
        }
        return root;
    }

    public int candy(int[] ratings) {
        int[] candies = new int[ratings.length];
        candies[0] = 1;
        for (int i = 1; i < ratings.length; i++) {
            if (ratings[i] > ratings[i - 1]) {
                candies[i] = candies[i - 1] + 1;
            } else if (ratings[i] == ratings[i - 1]) {
                candies[i] = candies[i - 1];
            } else {
                int j = i + 1;
                int count = 1;
                while (j < ratings.length && ratings[j] < ratings[j - 1]) {
                    j++;
                    count++;
                }
                if (count >= candies[i - 1]) {
                    candies[i - 1] = count + 1;
                }
                for (int m = i; m < j; m++) {
                    candies[m] = count;
                    count--;
                }
                i = j - 1;
            }
        }
        int sum = 0;
        for (int c : candies) {
            sum += c;
        }
        return sum;
    }

    public int findMin(int[] nums) {
        int start = 0, stop = nums.length - 1;
        while (start < stop - 1) {
            if (nums[start] < nums[stop]) {
                return nums[start];
            }
            int mid = start + (stop - start) / 2;
            if (nums[start] < nums[mid]) {
                start = mid;
            } else if (nums[start] > nums[mid]) {
                stop = mid;
            } else {
                start++;
            }
        }
        return Math.min(nums[start], nums[stop]);
    }

    public int largestRectangleArea(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int[] heightNew = new int[heights.length + 1];
        for (int i = 0; i < heights.length; i++) {
            heightNew[i] = heights[i];
        }
        int sum = 0;
        int i = 0;
        while (i < heightNew.length) {
            if (stack.isEmpty() || heightNew[i] > heightNew[stack.peek()]) {
                stack.push(i);
                i++;
            } else {
                int ln = stack.pop();
                sum = Math.max(sum, heightNew[ln] * (stack.isEmpty() ? i : i - stack.peek() - 1));
            }
        }
        return sum;
    }

    public int maximalRectangle(char[][] matrix) {
        if(matrix.length==0)return 0;
        int m = matrix.length, n = matrix[0].length;
        int[][] heightTable = new int[m][n + 1];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '0') {
                    heightTable[i][j] = 0;
                } else {
                    heightTable[i][j] = i == 0 ? 1 : heightTable[i - 1][j] + 1;
                }
            }
        }
        int maxArea = 0;
        for (int i = 0; i < m; i++) {
            maxArea = Math.max(maxArea, largestRectangleAreaHelper(heightTable[i]));
        }
        return maxArea;
    }

    private int largestRectangleAreaHelper(int[] height) {
        Stack<Integer> stack = new Stack<>();
        int i = 0, max = 0;
        while (i < height.length) {
            if (stack.isEmpty() || height[i] > height[stack.peek()]) {
                stack.push(i);
                i++;
            } else {
                int cur = stack.pop();
                max = Math.max(max, height[cur] * (stack.isEmpty() ? i : i - stack.peek() - 1));
            }
        }
        return max;
    }

    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> result=new ArrayList<>();
        if(root==null) return result;
        Stack<TreeNode> stack=new Stack<>();
        stack.push(root);
        TreeNode pre=null;
        while(!stack.isEmpty()){
            TreeNode node=stack.peek();
            if((node.left==null&&node.right==null)||(pre!=null&&(pre==node.left||pre==node.right))){
                stack.pop();
                result.add(node.val);
                pre=node;
            }else{
                if(node.right!=null){
                    stack.push(node.right);
                }
                if(node.left!=null){
                    stack.push(node.left);
                }
            }
        }
        return result;
    }

    public void recoverTree(TreeNode root) {

    }

    public static void main(String[] args) {
        System.out.println(new Hard().candy(new int[]{1, 2, 2}));
    }
}
