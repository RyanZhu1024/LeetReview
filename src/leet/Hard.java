package leet;


import java.util.*;

/**
 * Created by RyanZhu on 1/20/16.
 */
public class Hard {

    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }

    }

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

    public int candyAnother(int[] ratings) {
        if (ratings == null || ratings.length == 0) return 0;
        int[] candies = new int[ratings.length];
        Arrays.fill(candies, 1);
        for (int i = 1; i < candies.length; i++) {
            if (ratings[i] > ratings[i - 1]) {
                candies[i] = candies[i - 1] + 1;
            }
        }
        for (int i = candies.length - 1; i >= 1; i--) {
            if (ratings[i - 1] > ratings[i] && candies[i - 1] <= candies[i]) {
                candies[i - 1] = candies[i] + 1;
            }
        }
        int sum = 0;
        for (int candy : candies) {
            sum += candy;
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
        if (matrix.length == 0) return 0;
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
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        TreeNode pre = null;
        while (!stack.isEmpty()) {
            TreeNode node = stack.peek();
            if ((node.left == null && node.right == null) || (pre != null && (pre == node.left || pre == node.right))) {
                stack.pop();
                result.add(node.val);
                pre = node;
            } else {
                if (node.right != null) {
                    stack.push(node.right);
                }
                if (node.left != null) {
                    stack.push(node.left);
                }
            }
        }
        return result;
    }

    public void recoverTree(TreeNode root) {
        if (root == null) return;
        TreeNode cur = root, pre = null, first = null, second = null, preCur = null;
        while (cur != null) {
            if (cur.left == null) {
                if (found(preCur, cur)) {
                    if (first == null) {
                        first = preCur;
                    }
                    second = cur;
                }
                preCur = cur;
                cur = cur.right;
            } else {
                pre = cur.left;
                while (pre.right != null && pre.right != cur) {
                    pre = pre.right;
                }
                if (pre.right == null) {
                    pre.right = cur;
                    cur = cur.left;
                } else if (pre.right == cur) {
                    if (found(preCur, cur)) {
                        if (first == null) {
                            first = preCur;
                        }
                        second = cur;
                    }
                    preCur = cur;
                    pre.right = null;
                    cur = cur.right;
                }
            }
        }
        if (first != null && second != null) {
            int temp = first.val;
            first.val = second.val;
            second.val = temp;
        }
    }

    public boolean found(TreeNode pre, TreeNode cur) {
        return pre != null && pre.val > cur.val;
    }

    public void morrisTraversal(TreeNode root) {
        if (root == null) return;
        TreeNode cur = root;
        TreeNode pre = null;
        while (cur != null) {
            if (cur.left == null) {
                System.out.println(cur.val);
                cur = cur.right;
            } else if (cur.left != null) {
                pre = findPre(cur);
                if (pre.right == null) {
                    pre.right = cur;
                    cur = cur.left;
                } else if (pre == cur) {
                    pre.right = null;
                    System.out.println(cur.val);
                    cur = cur.right;
                }
            }
        }
    }

    public TreeNode findPre(TreeNode node) {
        TreeNode temp = node.left;
        while (temp.right != null) {
            temp = temp.right;
        }
        return temp;
    }

    public int maxProfit(int[] prices) {
        if (prices == null || prices.length <= 1) return 0;
        int minP = prices[0], sum = 0;
        int[] table = new int[prices.length];
        for (int i = 1; i < prices.length; i++) {
            minP = Math.min(minP, prices[i - 1]);
            sum = Math.max(sum, prices[i] - minP);
            table[i] = sum;
        }
        int maxS = prices[prices.length - 1], max2 = Integer.MIN_VALUE;
        for (int i = prices.length - 2; i >= 0; i--) {
            maxS = Math.max(maxS, prices[i + 1]);
            max2 = Math.max(max2, maxS - prices[i]);
            if (max2 > 0) {
                table[i] = table[i] + max2;
                sum = Math.max(sum, table[i]);
            }
        }
        return sum > 0 ? sum : 0;
    }

    public int jump(int[] nums) {
        int i = 0, steps = 0, cur = 0, next = 0;
        while (i < nums.length) {
            if (cur >= nums.length - 1) break;
            while (i <= cur) {
                next = Math.max(next, nums[i] + i);
                i++;
            }
            steps++;
            cur = next;
        }
        return steps;
    }

    public List<String> wordBreak2(String s, Set<String> wordDict) {
        //TODO
        return null;
    }

    public List<String> wordBreak(String s, Set<String> wordDict) {
        List<Integer>[] starts = new List[s.length() + 1]; // valid start positions
        starts[0] = new ArrayList<Integer>();

        int maxLen = getMaxLen(wordDict);
        for (int i = 1; i <= s.length(); i++) {
            for (int j = i - 1; j >= i - maxLen && j >= 0; j--) {
                if (starts[j] == null) continue;
                String word = s.substring(j, i);
                if (wordDict.contains(word)) {
                    if (starts[i] == null) {
                        starts[i] = new ArrayList<Integer>();
                    }
                    starts[i].add(j);
                }
            }
        }

        List<String> rst = new ArrayList<>();
        if (starts[s.length()] == null) {
            return rst;
        }

        dfs(rst, "", s, starts, s.length());
        return rst;
    }


    private void dfs(List<String> rst, String path, String s, List<Integer>[] starts, int end) {
        if (end == 0) {
            rst.add(path.substring(1));
            return;
        }

        for (Integer start : starts[end]) {
            String word = s.substring(start, end);
            dfs(rst, " " + word + path, s, starts, start);
        }
    }

    private int getMaxLen(Set<String> wordDict) {
        int max = 0;
        for (String s : wordDict) {
            max = Math.max(max, s.length());
        }
        return max;
    }

    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        int n=lists.length;
        int left=0,right=n-1;
        while(right>0) {
            while (left < right) {
                lists[left] = merge(lists[left], lists[right]);
                left++;
                right--;
            }
            left = 0;
        }
        return lists[0];
    }

    public ListNode merge(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(Integer.MIN_VALUE);
        ListNode node = head;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                node.next = l1;
                l1 = l1.next;
                node = node.next;
            } else {
                node.next = l2;
                l2 = l2.next;
                node = node.next;
            }
        }
        if(l1!=null){
            node.next=l1;
        }else{
            node.next=l2;
        }
        return head.next;
    }

    public static void main(String[] args) {
        Hard h = new Hard();
        String s = "aaaaaaaaaaaa";
        Set<String> set = new HashSet<>();
        set.add("a");
        set.add("aaa");
        set.add("aa");
        set.add("aaaa");
        set.add("aaaaa");
        System.out.println(h.wordBreak(s, set));
    }
}
