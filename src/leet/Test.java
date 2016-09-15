package leet;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
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

    public static class NumArray {

        int[] tree;
        int[] nums;

        public NumArray(int[] nums) {
            this.tree = new int[nums.length + 1];
            this.nums = nums;
            for (int i = 1; i < nums.length + 1; i++) {
                updateTree(i, nums[i - 1]);
            }
        }

        private void updateTree(int index, int val) {
            while (index < tree.length) {
                tree[index] += val;
                index = getNext(index);
            }
        }

        private int getNext(int i) {
            return i + (i & -i);
        }

        private int getParent(int i) {
            return i - (i & -i);
        }

        void update(int i, int val) {
            int diff = val - nums[i];
            nums[i] = val;
            updateTree(i + 1, diff);
        }

        public int sumRange(int i, int j) {
            int sum = 0;
            j++;
            while (j > i) {
                sum += tree[j];
                j = getParent(j);
            }
            return sum;
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

    public static void main(String[] args) throws IOException {
    }

    static void pingip() throws IOException {
        String[] ips = new String[]{"8.8.8.8"};
        Runtime runtime = Runtime.getRuntime();
        int counts = 3;
        for (String ip : ips) {
            double sumsd = 0, summean = 0;
            for (int j = 0; j < counts; j++) {
                Process process = runtime.exec("ping -c 4 " + ip);
                BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                String inputLine;
                List<String> lists = new ArrayList<>();
                while ((inputLine = reader.readLine()) != null) {
                    lists.add(inputLine);
                    System.out.println(inputLine);
                }
                lists = lists.subList(1, 5);
                double[] times = new double[4];
                for (int i = 0; i < 4; i++) {
                    String res = lists.get(i);
                    String[] group = res.split(" ");
                    if (group.length > 6) {
                        String s = group[6].trim();
                        times[i] = Double.valueOf(s.substring(s.indexOf("=") + 1));
                    }
                }
                sumsd += sd(times);
                summean += avg(times);
                reader.close();
            }
            System.out.println("---------------------------");
            System.out.println("IP: " + ip);
            System.out.println(String.format("ST DEV: %.3f", sumsd / counts));
            System.out.println(String.format("ST MEAN: %.3f", summean / counts));
            System.out.println("---------------------------");
        }
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

    static double avg(double[] arr) {
        return Arrays.stream(arr).average().getAsDouble();
    }

    static double sd(double[] arr) {
        double avg = Arrays.stream(arr).average().getAsDouble();
        final double[] sum = {0};
        Arrays.stream(arr).forEach(value -> sum[0] += Math.pow(value - avg, 2));
        double res = Math.sqrt(sum[0] / (arr.length));
        return res;
    }
}