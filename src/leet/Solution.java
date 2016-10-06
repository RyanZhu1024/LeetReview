package leet;

import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Created by RyanZhu on 28/09/2016.
 */
public class Solution {

    class Range {
        int begin;
        int end;
        public Range(int i, int j) {
            begin = i;
            end = j;
        }
    }

    class Wrapper {
        int num;
        int index;
        public Wrapper(int i, int j) {
            num = i;
            index = j;
        }
    }

    Wrapper[] wrappers;

    Map<Integer, Range> map = new HashMap<>();
    public Solution(Integer[] nums) {
        wrappers = new Wrapper[nums.length];
        for (int i = 0; i < nums.length; i++) {
            wrappers[i] = new Wrapper(nums[i], i);
        }
        Arrays.sort(wrappers,(w1, w2) -> w1.num - w2.num);

        for (int i = 0; i < wrappers.length; i++) {
            if (map.containsKey(wrappers[i].num)) {
                map.get(wrappers[i].num).end = i;
            } else {
                map.put(wrappers[i].num, new Range(i, i));
            }
        }
    }

    public Solution() {
    }

    public int pick(int target) {
        Range range = map.get(target);
        int bound = range.end - range.begin + 1;
        Random r = new Random();
        return wrappers[r.nextInt(bound) + range.begin].index;
    }

    public static void main(String[] args) throws FileNotFoundException {
        Solution s = new Solution();
        System.out.println(s.minWindow("abc","a"));
    }

    public int maxSubArray(int[] nums, int k) {
        // write your code here
        if (nums == null || nums.length < k || k == 0) {
            return 0;
        }
        int n = nums.length;
        int[][] dp = new int[n + 1][k + 1];
        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < k + 1; j++) {
                int maxEndsHere = 0, maxSoFar = Integer.MIN_VALUE, max = Integer.MIN_VALUE;
                for (int p = i; p >= j; p--) {
                    maxEndsHere = Math.max(nums[p - 1], maxEndsHere + nums[p - 1]);
                    maxSoFar = Math.max(maxSoFar, maxEndsHere);
                    max = Math.max(dp[p - 1][j - 1] + maxSoFar, max);
                }
                dp[i][j] = max;
            }
        }
        return dp[n][k];
    }

    public String canReach(int a, int b, int c, int d) {
        return helper(a, b, c, d) ? "Yes" : "No";
    }

    public boolean helper(int a, int b, int c, int d) {
        if (a > c || b > d) return false;
        if (a == c && b == d) return true;
        return helper(a, a + b, c, d) || helper(a + b, b, c, d);
    }

    public int change(int n, int... pointing) {
        boolean[] goods = new boolean[n + 1];
        boolean[] visited = new boolean[n + 1];
        visited[1] = true;
        goods[1] = true;
        int sum = 0;
        for (int i = 1; i <= n; i++) {
            if (visited[i]) continue;
            sum += dfs(i, pointing, goods,visited);
        }
        return sum;
    }

    int dfs(int cur, int[] pointing, boolean[] goods, boolean[] visited) {
        if (cur == 1) return 0;
        if (goods[cur]) return 0;
        if (visited[cur]) return 1;
        visited[cur] = true;
        int change = dfs(pointing[cur - 1], pointing, goods, visited);
        goods[cur] = true;
        return change;
    }

    public String minWindow(String source, String target) {
        // write your code
        if (target == null || target.isEmpty()) return target;
        int i = 0, j = 0;
        Map<Character, Integer> map = new HashMap<>();
        for (char c : target.toCharArray()) {
            if (!map.containsKey(c)) {
                map.put(c, 0);
            }
            map.put(c, map.get(c) + 1);
        }
        int count = target.length(), len = Integer.MAX_VALUE;
        String res = "";
        while (j < source.length()) {
            char cur = source.charAt(j);
            if (map.containsKey(cur)) {
                if (map.get(cur) > 0) {
                    count--;
                }
                map.put(cur, map.get(cur) - 1);
                while (count == 0) {
                    char start = source.charAt(i);
                    if (j - i + 1 < len) {
                        len = j - i + 1;
                        res = source.substring(i, j + 1);
                    }
                    if (map.containsKey(start)) {
                        if (map.get(start) == 0) {
                            count++;
                        }
                        map.put(start, map.get(start) + 1);
                    }
                    i++;
                }
            }
            j++;
        }
        return res;
    }
}
