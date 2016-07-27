package leet;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

/**
 * Created by rzhu on 7/5/16.
 */
public class Facebook {
    public static void main(String[] args) {
        Facebook facebook = new Facebook();
        System.out.println(facebook.numDecodings("11"));
    }

    public int numDecodings(String s) {
        if (s == null || s.isEmpty()) return 0;
        int[] dp = new int[s.length() + 1];
        dp[0] = 1;
        dp[1] = Character.getNumericValue(s.charAt(0)) == 0 ? 0 : 1;
        for (int i = 1; i < s.length(); i++) {
            int cur = Character.getNumericValue(s.charAt(i));
            if (cur == 0) {
                dp[i + 1] = dp[i - 1];
            } else {
                dp[i + 1] = dp[i];
                int pre = Character.getNumericValue(s.charAt(i - 1));
                int com = pre * 10 + cur;
                if (com > 10 && com <= 26) {
                    dp[i + 1]++;
                }
            }
        }
        return dp[s.length()];
    }

    public String minWindowRefined(String s, String t) {
        Map<Character, Integer> map = new HashMap<>();
        for (char c : t.toCharArray()) {
            if (map.containsKey(c)) {
                map.put(c, map.get(c) + 1);
            } else {
                map.put(c, 1);
            }
        }
        int counter = t.length(), begin = 0, end = 0, len = Integer.MAX_VALUE;
        String result = "";
        while (end < s.length()) {
            char cur = s.charAt(end);
            if (map.containsKey(cur)) {
                if (map.get(cur) > 0) counter--;
                map.put(cur, map.get(cur) - 1);
                while (counter == 0) {
                    char start = s.charAt(begin);
                    if (map.containsKey(start)) {
                        if (end - begin + 1 < len) {
                            len = end - begin + 1;
                            result = s.substring(begin, end + 1);
                        }
                        if (map.get(start) == 0) {
                            counter++;
                        }
                        map.put(start, map.get(start) + 1);
                    }
                    begin++;
                }
            }
            end++;
        }
        return result;
    }

    public String minWindow(String s, String t) {
        if (s == null || t == null || t.length() > s.length()) {
            return "";
        } else {
            Map<Character, Integer> map = new HashMap<>();
            for (char c : t.toCharArray()) {
                if (map.containsKey(c)) {
                    map.put(c, map.get(c) + 1);
                } else {
                    map.put(c, 1);
                }
            }
            LinkedList<Item> queue = new LinkedList<>();
            int count = 0, len = Integer.MAX_VALUE;
            String result = "";
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (map.containsKey(c)) {
                    if (map.get(c) > 0 && count < t.length()) {
                        count++;
                    }
                    map.put(c, map.get(c) - 1);
                    queue.offer(new Item(i, c));
                    if (count == t.length()) {
                        while (map.get(queue.peek().val) < 0) {
                            char key = queue.poll().val;
                            map.put(key, map.get(key) + 1);
                        }
                        int start = queue.peek().index, end = queue.peekLast().index;
                        if (end - start + 1 < len) {
                            len = end - start + 1;
                            result = s.substring(start, end + 1);
                        }
                    }
                }
            }
            if (count == t.length()) {
                while (map.get(queue.peek().val) < 0) {
                    queue.poll();
                }
                int start = queue.peek().index, end = queue.peekLast().index;
                if (end - start + 1 < len) {
                    len = end - start + 1;
                    result = s.substring(start, end + 1);
                }
            }
            return result;
        }
    }

    class Item {
        int index;
        char val;
        public Item(int i, char c) {
            this.index = i;
            this.val = c;
        }
    }
    public int combine123to100() {
        int[] coins = new int[]{1,2,5};
        int n = 100;
        return combine123to100Helper(coins, n, 0);
    }

    public int combine123to100DP() {
        int[] coins = new int[]{1,2,5};
        int n = 100;
//        int[] dp = new int[101];
//        dp[0] = 1;
//        for(int i = 0; i < coins.length; i++) {
//            for(int j = coins[i]; j <= 100; j++) {
//                dp[j] += dp[j - coins[i]];
//            }
//        }
//        return dp[100];
        int[][] dp = new int[4][101];
        for(int i = 0; i < 4; i++) dp[i][0] = 1;
        for(int i = 1; i <= 3; i++) {
            for(int j = 1;j <= 100; j++) {
                dp[i][j] = dp[i - 1][j];
                if(j - coins[i - 1] >= 0) {
                    dp[i][j] += dp[i][j - coins[i - 1]];
                }
            }
        }
        return dp[3][100];
    }

    private int combine123to100Helper(int[] coins, int n, int begin) {
        if(n == 0) return 1;
        if(n < 0) return 0;
        else {
            int sum = 0;
            for(int i = begin; i < coins.length; i++) {
                sum += combine123to100Helper(coins, n - coins[i], i);
            }
            return sum;
        }
    }

    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        } else {
            int m = grid.length, n = grid[0].length;
            boolean[][] table = new boolean[m][n];
            int sum = 0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (!table[i][j] && grid[i][j] == '1') {
                        helper(i, j, grid, table);
                        sum++;
                    }
                }
            }
            return sum;
        }
    }

    void helper(int i, int j, char[][] grid, boolean[][] table) {//maze, // visit
        int m = grid.length, n = grid[0].length;
        if (grid[i][j] == '1' && !table[i][j]) {
            table[i][j] = true;
            //left
            if (j > 0) {
                helper(i, j - 1, grid, table);
            }
            //right
            if (j < n - 1) {
                helper(i, j + 1, grid, table);
            }
            //up
            if (i > 0) {
                helper(i - 1, j, grid, table);
            }
            //down
            if (i < m - 1) {
                helper(i + 1, j, grid, table);
            }
        }
    }

    public int divide(int dividend, int divisor) {
        if (divisor == 0) {
            return Integer.MAX_VALUE;
        }
        int count = 0, negative = 1;
        if (dividend > 0 && divisor < 0) {
            negative = -1;
            divisor = -divisor;
        }
        if (dividend < 0 && divisor > 0) {
            negative = -1;
            dividend = -dividend;
        }
        if (dividend < 0 && divisor < 0) {
            dividend = -dividend;
            divisor = -divisor;
        }
        while (dividend >= divisor) {
            dividend -= divisor;
            count++;
            if (count == Integer.MAX_VALUE) {
                break;
            }
        }
        return count * negative;
    }
}
