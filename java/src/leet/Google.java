package leet;

import java.util.*;

/**
 * Created by rzhu on 7/28/16.
 */
public class Google {
    public static void main(String[] args) {
        Google g = new Google();
        Set<String> set = new HashSet<>();
        set.add("a");
        set.add("aa");
        set.add("aaa");
        set.add("aaaa");
        set.add("aaaaa");
        set.add("aaaaaa");
        set.add("aaaaaaa");
        set.add("aaaaaaaa");
        set.add("aaaaaaaaa");
        set.add("aaaaaaaa");
        g.wordBreak2("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", set);
    }

    public List<String> wordBreak2(String s, Set<String> wordDict) {
        List<String> res = new ArrayList<>();
        if (s == null || s.isEmpty() || wordDict.isEmpty()) {
            return res;
        }
        int len = getMaxLen(wordDict);
        List<Integer>[] indices = wordBreakHelper(s, wordDict, len);
        dfs(indices, s, wordDict, res, 0, "");
        return res;
    }

    void dfs(List<Integer>[] indices, String s, Set<String> wordDict, List<String> res, int start, String cur) {
        if (indices[s.length()] == null) return;
        if (start == s.length() && !cur.isEmpty()) {
            res.add(cur.trim());
            return;
        }
        for (int index : indices[start]) {
            String sub = s.substring(start, index);
            dfs(indices, s, wordDict, res, index, cur + sub + " ");
        }
    }

    List<Integer>[] wordBreakHelper(String s, Set<String> wordDict, int len) {
        List<Integer>[] dp = new List[s.length() + 1];
        dp[0] = new ArrayList<>();
        for (int i = 1; i <= s.length(); i++) {
            for (int j = i - 1; j >= 0 && i - j <= len; j--) {
                if (dp[j] != null) {
                    String sub = s.substring(j, i);
                    if (wordDict.contains(sub)) {
                        dp[j].add(i);
                        if (dp[i] == null) {
                            dp[i] = new ArrayList<>();
                        }
                    }
                }
            }
        }
        return dp;
    }

    int getMaxLen(Set<String> wordDict) {
        int max = 0;
        for (String word : wordDict) {
            max = Math.max(max, word.length());
        }
        return max;
    }

    public void arrangeCoins(long[] coins) {
        for (int i = 0; i < coins.length; i++) {
            long coin = coins[i];
            int cur = bisearch(coin);
            System.out.println(cur);
        }
    }

    private long getCoin(int c) {
        long t = (long) c;
        return ((1 + t) * t) / 2;
    }

    private int bisearch(long coin) {
        int left = 1, right = Integer.MAX_VALUE;
        while (left + 1 < right) {
            int mid = left + (right - left) / 2;
            long tc = getCoin(mid);
            if (tc == coin) {
                return mid;
            } else if (tc < coin) {
                left = mid;
            } else {
                right = mid;
            }
        }
        if (getCoin(right) < coin) return right;
        return left;
    }

    public int trap(int[] height) {
        if (height == null || height.length < 3) {
            return 0;
        }
        Stack<Integer> st = new Stack<>();
        int i = 0, sum = 0;
        while (i < height.length) {
            // while (st.isEmpty() || height[i] <= height[st.peek()]) {
            //     st.push(i);
            //     i++;
            // }
            while (!st.isEmpty() && height[i] > height[st.peek()]) {
                int top = height[st.pop()];
                int j = st.isEmpty() ? -1 : st.peek();
                while (!st.isEmpty() && height[st.peek()] == top) {
                    j = st.pop();
                }
                if (j > 0 && height[st.peek()] > top) {
                    sum += (Math.min(height[st.peek()], height[i]) - top) * ((i - st.peek()) - 1);
                }
            }
            st.push(i);
            i++;
        }
        return sum;
    }

    class BinaryIndexTree {
        private int[] tree;

        public BinaryIndexTree(int[] input) {
            tree = new int[input.length + 1];
            for (int i = 1; i < input.length + 1; i++) {
                updateTree(input[i], i);
            }
        }

        public int getPreSum(int index) {
            int sum = 0;
            index++;
            while (index > 0) {
                sum += tree[index];
                index = getParent(index);
            }
            return sum;
        }

        private void updateTree(int val, int index) {
            while (index < tree.length) {
                tree[index] += val;
                index = getNext(index);
            }
        }

        private int getParent(int index) {
            return index - (index & -index);
        }
        private int getNext(int index) {
            return index + (index & -index);
        }
    }

    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= (int)Math.sqrt(i); j++) {
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }


    public ArrayList<ArrayList<Integer>> buildingOutline(int[][] buildings) {
        // write your code here
        if (buildings == null || buildings.length == 0) {
            return new ArrayList<>();
        }
        List<Point> points = new ArrayList<>();
        for (int i = 0; i < buildings.length;i++) {
            points.add(new Point(buildings[i][0], buildings[i][2]));
            points.add(new Point(buildings[i][1], -buildings[i][2]));
        }
        Collections.sort(points, new Comparator<Point>(){
            @Override
            public int compare(Point p1, Point p2) {
                return p1.x - p2.x;
            }
        });
        for (int i = 0; i < buildings.length;i++) {
            int start = buildings[i][0], end = buildings[i][1];
            for (Point p : points) {
                if (p.x >= start && p.x < end) {
                    p.y = Math.max(p.y, buildings[i][2]);
                }
            }
        }
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        Point pre = points.get(0);
        int i = 1;
        while (i < points.size()) {
            Point cur = points.get(i);
            if (pre.y < 0) {
                pre = cur;
            } else if (cur.y != pre.y) {
                ArrayList<Integer> triple = new ArrayList<>();
                triple.add(pre.x);
                triple.add(cur.x);
                triple.add(pre.y);
                res.add(triple);
                pre = cur;
            }
            i++;
        }
        return res;
    }

    class Point {
        int x;
        int y;
        public Point (int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int m = nums1.length, n = nums2.length;
        List<Integer>[][] predp = new List[m + 1][n + 1];
        List<Integer>[][] dp = new List[m + 1][n + 1];
        for (int i = 0; i < m + 1; i++) {
            for (int j = 0; j < n + 1; j++) {
                predp[i][j] = new ArrayList<>();
            }
        }
        for (int o = 1; o <= k; o++) {
            for (int i = o; i <= n; i++) {
                List<Integer> list = new ArrayList<>(predp[0][i - 1]);
                list.add(nums2[i - 1]);
                dp[0][i] = max(dp[0][i - 1], list);
            }
            for (int i = o; i <= m; i++) {
                List<Integer> list = new ArrayList<>(predp[i - 1][0]);
                list.add(nums1[i - 1]);
                dp[i][0] = max(dp[i - 1][0], list);
            }
            for (int i = 1; i < m + 1; i++) {
                for (int j = 1; j < n + 1; j++) {
                    if (i + j < o) continue;
                    List<Integer> temp1 = new ArrayList<>(predp[i - 1][j]);
                    temp1.add(nums1[i - 1]);
                    List<Integer> temp2 = new ArrayList<>(predp[i][j - 1]);
                    temp2.add(nums2[j - 1]);

                    dp[i][j] = max(max(dp[i - 1][j], dp[i][j - 1]), max(temp1, temp2));
                }
            }
            predp = dp;
            dp = new List[m + 1][n + 1];
        }
        int[] res = new int[k];
        for (int i = 0; i < k; i++) {
            res[i] = predp[m][n].get(i);
        }
        return res;
    }

    List<Integer> max(List<Integer> l1, List<Integer> l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        if(l1.size() > l2.size()) {
            return l1;
        } else if (l1.size() < l2.size()) {
            return l2;
        } else {
            for (int i = 0; i < l1.size(); i++) {
                if (l1.get(i) > l2.get(i)) {
                    return l1;
                } else if (l1.get(i) < l2.get(i)) {
                    return l2;
                }
            }
            return l1;
        }
    }

    public boolean validTree(int n, int[][] edges) {
        // Write your code here
        if (n == 0 || edges == null || edges.length == 0) {
            return true;
        }
        Map<Integer, ArrayList<Integer>> map = new HashMap<>();
        for (int i = 0; i < edges.length; i++) {
            Integer n1 = edges[i][0], n2 = edges[i][1];
            if (!map.containsKey(n1)) {
                map.put(n1, new ArrayList<Integer>());
            }
            if (!map.containsKey(n2)) {
                map.put(n2, new ArrayList<Integer>());
            }
            map.get(n1).add(n2);
            map.get(n2).add(n1);
        }
        Set<Integer> set = new HashSet<>();
        int node = edges[0][0];
        set.add(node);
        return dfs(map, set, node, Integer.MIN_VALUE) && set.size() == n;
    }

    boolean dfs(Map<Integer, ArrayList<Integer>> map, Set<Integer> set, int node, int parent) {
        boolean result = true;
        ArrayList<Integer> neighbors = map.get(node);
        for (Integer nei : neighbors) {
            if (nei != parent) {
                if (set.contains(nei)) {
                    return false;
                }
                set.add(nei);
                result = result && dfs(map, set, nei, node);
            }
        }
        return result;
    }

    public boolean regularExpressionMatching(String s, String p) {
        // write your code here
        if (s == null || p == null) {
            return false;
        } else {
            int m = s.length(), n = p.length();
            boolean[][] dp = new boolean[m + 1][n + 1];
            dp[0][0] = true;
            for (int i = 1; i < m + 1; i++) {
                dp[i][0] = false;
            }
            for (int i = 1; i < n + 1; i++) {
                if (p.charAt(i - 1) == '*') {
                    dp[0][i] = dp[0][i - 2];
                }
            }
            for (int i = 1; i < m + 1; i++) {
                for (int j = 1; j < n + 1; j++) {
                    if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                        dp[i][j] = dp[i - 1][j - 1];
                    } else if (p.charAt(j - 1) == '*') {
                        dp[i][j] = (dp[i - 1][j - 1] && s.charAt(i - 1) == p.charAt(j - 2) || p.charAt(j - 2) == '.') || dp[i][j - 1] || dp[i][j - 2];
                    }
                }
            }
            return dp[m][n];
        }
    }

    public int nthSuperUglyNumber(int n, int[] primes) {
        // Write your code here
        int count = 1;
        Map<Integer, Integer> map = new HashMap<>();
        for (int p : primes) {
            map.put(p, 0);
        }
        int[] res = new int[n];
        res[0] = 1;
        while (count < n) {
            int minVal = Integer.MAX_VALUE, minKey = -1;
            for (int p : primes) {
                if (res[map.get(p)] * p < minVal) {
                    minVal = res[map.get(p)] * p;
                    minKey = p;
                }
            }
            map.put(minKey, map.get(minKey) + 1);
            if (minVal > res[count - 1]) {
                res[count] = minVal;
                count++;
            }
        }
        return res[n - 1];
    }

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

    public int findMaxOA11(int num) {
        // delete one digit from identical adjacent digits
        int max = 0;
        String numStr = String.valueOf(num);
        int i = 0;
        while (i < numStr.length() - 1) {
            int j = i;
            while (j < numStr.length() - 1 && numStr.charAt(j + 1) == numStr.charAt(i)) {
                j++;
            }
            if (j > i) {
                String p1 = numStr.substring(0, i);
                String p2 = numStr.substring(i, j);
                String p3 = numStr.substring(j + 1);
                max = Math.max(max, Integer.parseInt(p1 + p2 + p3));
            }
            i = j + 1;
        }
        return max;
    }

    public int findMaxOA12(int num) {
        // repeat one digit and get the max value
        if (num == 0) {
            return 0;
        } else {
            String numstr = String.valueOf(num);
            int max = 0;
            for (int i = 0; i < numstr.length(); i++) {
                if (i == numstr.length() - 1) {
                    max = Integer.parseInt(numstr + numstr.charAt(i));
                } else if (numstr.charAt(i) > numstr.charAt(i + 1)) {
                    max = Integer.parseInt(numstr.substring(0, i + 1) +
                            numstr.charAt(i) + numstr.substring(i + 1));
                    break;
                }
            }
            return max;
        }
    }

    public int findMaxOA13(int num) {
        //delete adjacent greater digit and get max
        if (num == 0) {
            return 0;
        } else {
            String numstr = String.valueOf(num);
            char preChar = numstr.charAt(0);
            int max = 0;
            for (int i = 1; i < numstr.length(); i++) {
                char curChar = numstr.charAt(i);
                if (curChar > preChar) {
                    String temp = numstr.substring(0, i) + numstr.substring(i + 1);
                    max = Math.max(max, Integer.parseInt(temp));
                } else {
                    String temp = numstr.substring(0, i - 1) + numstr.substring(i);
                    max = Math.max(max, Integer.parseInt(temp));
                }
                preChar = curChar;
            }
            return max;
        }
    }

    private class FileTreeNode {
        String name;
        List<FileTreeNode> children;
        FileTreeNode parent;
        boolean isPicture;
        int spaces;

        FileTreeNode(String n, boolean p, FileTreeNode pa, int sp) {
            children = new ArrayList<>();
            name = n.trim();
            isPicture = p;
            parent = pa;
            spaces = sp;
        }
    }

    int solution(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        } else {
            String[] paths = s.split("\n");
            FileTreeNode root = buildFileTree(paths);
            List<String> result = new ArrayList<>();
            dfs(root, "", result);
            System.out.println(result);
            return result.size();
        }
    }

    private void dfs(FileTreeNode node, String path, List<String> result) {
        if (node.isPicture) {
            result.add(path);
        } else {
            for (FileTreeNode child : node.children) {
                dfs(child, path + "/" + child.name, result);
            }
        }
    }

    private FileTreeNode buildFileTree(String[] paths) {
        FileTreeNode root = new FileTreeNode("", false, null, -1);
        FileTreeNode cur = root;
        for (String path : paths) {
            if (path.lastIndexOf(".") != -1 && !isPictureFile(path)) {
                continue;
            }
            int numberOfSpace = getNumberOfSpace(path);
            while (numberOfSpace < cur.spaces + 1) {
                cur = cur.parent;
            }
            FileTreeNode node = new FileTreeNode(path, isPictureFile(path), cur, numberOfSpace);
            cur.children.add(node);
            cur = node;
        }
        return root;
    }

    private boolean isPictureFile(String name) {
        return name.endsWith(".jpeg") || name.endsWith(".png") || name.endsWith(".gif");
    }

    private int getNumberOfSpace(String path) {
        int i = 0;
        while (i < path.length() && path.charAt(i) == ' ') {
            i++;
        }
        return i;
    }
    public void wiggleSort(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int i = 0, j = 0;
        boolean small = true;
        while (i < nums.length && j < nums.length) {
            while (j < nums.length && nums[j] == nums[i]){j++;}
            if (j < nums.length && ((small && nums[i] > nums[j]) || (!small && nums[i] < nums[j]))){
                swap(nums, i, j);
                small = !small;
            }
            i++;
        }
    }

    void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
