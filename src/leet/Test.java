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

    public ArrayList<ArrayList<Integer>> subsets(int[] nums) {
        // write your code here
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if(nums == null || nums.length == 0){
            return result;
        }
        Arrays.sort(nums);
        ArrayList<Integer> cur = new ArrayList<>();
        helper(nums, 0, cur, result);
        return result;
    }

    void helper(int[] nums, int start, ArrayList<Integer> cur, ArrayList<ArrayList<Integer>> result){
        result.add(new ArrayList<>(cur));
        for(int i = start;i < nums.length;i++){
            cur.add(nums[i]);
            helper(nums, i + 1, cur, result);
            cur.remove(cur.size() - 1);
        }
    }

    public static void main(String[] args) {
        System.out.println(new Test().numDecodings("192611"));
    }

    public int numDecodings(String s) {
        // Write your code here
        if (s == null || s.isEmpty() || s.charAt(0) == '0') return 0;
        int[] dp = new int[s.length()];
        dp[0] = 1;
        for(int i = 1; i < s.length(); i++) {
            dp[i] = dp[i - 1];
            if (s.charAt(i) == '0') {
                dp[i] = Math.max(1, dp[i - 1] - 1);
            } else {
                char pc = s.charAt(i - 1);
                char c = s.charAt(i);
                if((pc == '1' && c > '0' && c <= '9') || (pc == '2' && c > '0' && c <= '6')) {
                    dp[i] *= 2;
                }
            }
        }
        return dp[s.length() - 1];
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
}