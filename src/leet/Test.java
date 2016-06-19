package leet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Queue;

public class Test {
    private static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    public static void main(String[] args) {
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        System.out.println(new Test().serialize(root));
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