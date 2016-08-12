package leet;

import java.util.*;

public class Test {
    private static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    public boolean isValidBST(TreeNode root) {
        // write your code here
        return helper(root, Integer.MAX_VALUE, Integer.MIN_VALUE);
    }

    boolean helper(TreeNode root, int max, int min){
        if(root == null){
            return true;
        }else{
            boolean left = true, right = true;
            if(root.left != null){
                if(root.left.val < root.val && root.left.val > min){
                    left = helper(root.left, root.val, min);
                }else{
                    left = false;
                }
            }
            if(root.right != null){
                if(root.right.val > root.val && root.right.val < max){
                    helper(root.right, max, root.val);
                }else{
                    right = false;
                }
            }
            return left && right;
        }
    }

    public List<List<Integer>> binaryTreePathSum(TreeNode root, int target) {
        // Write your code here
        List<List<Integer>> result = new ArrayList<>();
        if(root == null){
            return result;
        }else{
            List<Integer> cur = new ArrayList<>();
            helper(root, cur, result, target);
            return result;
        }
    }

    void helper(TreeNode root, List<Integer> cur, List<List<Integer>> result, int target){
        if(target == 0){
            result.add(new ArrayList<>(cur));
        }else{
            if(root == null){
                return;
            }
            cur.add(root.val);
            helper(root.left, new ArrayList<>(cur), result, target - root.val);
            helper(root.right, new ArrayList<>(cur), result, target - root.val);
        }
    }

    public static void main(String[] args) {
    }

    public ArrayList<ArrayList<Integer>> zigzagLevelOrder(TreeNode root) {
        // write your code here
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        } else {
            TreeNode dummy = new TreeNode(Integer.MIN_VALUE);
            LinkedList<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            queue.offer(dummy);
            ArrayList<Integer> cur = new ArrayList<>();
            boolean reverse = false;
            while (!queue.isEmpty()) {
                TreeNode node = queue.poll();
                if (node == dummy) {
                    if (reverse) {
                        Collections.reverse(cur);
                    }
                    result.add(cur);
                    if(!queue.isEmpty()) {
                        queue.offer(dummy);
                    }
                    cur = new ArrayList<>();
                    reverse = !reverse;

                } else {
                    cur.add(node.val);
                    if (node.left != null) {
                        queue.offer(node.left);
                    }
                    if (node.right != null) {
                        queue.offer(node.right);
                    }
                }
            }
            return result;
        }
    }

    /**
     * @param A      an integer array
     * @param target an integer
     * @param k      a non-negative integer
     * @return an integer array
     */
    public int[] kClosestNumbers(int[] A, int target, int k) {
        // Write your code here
        if (A == null || A.length == 0 || k > A.length) {
            return new int[0];
        } else {
            int left = 0, right = A.length - 1;
            while (left + 1 < right) {
                int mid = left + (right - left) / 2;
                if (A[mid] == target) {
                    left = mid;
                    right = mid;
                    break;
                } else if (A[mid] > target) {
                    right = mid;
                } else {
                    left = mid;
                }
            }
            int min = Integer.MAX_VALUE;
            LinkedList<Integer> list = new LinkedList<>();
            for (int i = 0; i < k; i++) {
                if (left == right) {
                    min = Math.min(min, Math.abs(A[left] - target));
                    list.add(A[left]);
                    left--;
                    right++;
                } else {
                    int leftV = left >= 0 ? A[left] : Integer.MAX_VALUE, rightV = right < A.length ? A[right] : Integer.MAX_VALUE;
                    int deltaLeft = Math.abs(leftV - target);
                    int deltaRight = Math.abs(rightV - target);
                    if (deltaLeft == deltaRight) {
                        i++;
                        if (deltaLeft == min) {
                            list.addFirst(leftV);
                            list.addLast(rightV);
                        } else {
                            list.addLast(leftV);
                            list.addLast(rightV);
                        }
                        min = deltaLeft;
                        left--;
                        right++;
                    } else if (deltaLeft < deltaRight) {
                        if (deltaLeft == min) {
                            list.addFirst(leftV);
                        } else {
                            list.addLast(leftV);
                        }
                        min = deltaLeft;
                        left--;
                    } else {
                        min = deltaRight;
                        list.addLast(rightV);
                        right++;
                    }
                }
            }
            int[] result = new int[k];
            for (int i = 0; i < k; i++) {
                result[i] = list.get(i);
            }
            return result;
        }
    }

    public String serialize(TreeNode root) {
        // write your code here
        ArrayList<String> strs = new ArrayList<>();
        if(root == null){
            return "";
        }
        Queue<TreeNode> queue = new LinkedList<>();
        TreeNode dummy = new TreeNode(Integer.MIN_VALUE);
        queue.offer(root);
        while(!queue.isEmpty()){
            TreeNode node = queue.poll();
            if(node == dummy){
                strs.add("#");
            }else{
                strs.add(String.valueOf(node.val));
                if(node.left != null){
                    queue.offer(node.left);
                }else{
                    queue.offer(dummy);
                }
                if(node.right != null){
                    queue.offer(node.right);
                }else{
                    queue.offer(dummy);
                }
            }
        }
        return String.join(",", strs);
    }

    public ArrayList<Integer> inorderTraversal(TreeNode root) {
        // write your code here
        ArrayList<Integer> result = new ArrayList<>();
        if(root == null){
            return result;
        }else{
            Stack<TreeNode> stack = new Stack<>();
            while(root != null){
                stack.push(root);
                root = root.left;
            }
            while(!stack.isEmpty()){
                TreeNode top = stack.pop();
                result.add(top.val);
                if(top.right != null){
                    stack.push(top.right);
                    TreeNode tmp = top.right.left;
                    while(tmp != null){
                        stack.push(tmp);
                    }
                }
            }
            return result;
        }
    }
}