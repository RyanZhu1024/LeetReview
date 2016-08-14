package leet;

import java.util.*;

/**
 * Created by rzhu on 7/28/16.
 */
public class Google {
    public static void main(String[] args) {
        Google g = new Google();
        System.out.println(g.isMatch("aa", ".*"));
    }

    public boolean isMatch(String s, String p) {
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
        System.out.println(g.nthSuperUglyNumber(11,new int[]{2,3,5}));
    }

    public int nthSuperUglyNumber(int n, int[] primes) {
        // Write your code here
        int count = 1;
        Map<Integer, Integer> map = new HashMap<>();
        for(int p : primes) {
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
                } else if (numstr.charAt(i) > numstr.charAt(i + 1)){
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

}
