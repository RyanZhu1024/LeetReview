package leet;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

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
        System.out.println(count("00110"));
    }

    static int count(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int i = 0;
        int count = 0;
        while (i < s.length() - 1) {
            int j = i + 1;
            while (j < s.length() && s.charAt(i) == s.charAt(j)) {
                j++;
            }
            int c1 = j - i;
            int nextStart = j;
            int k = j + 1;
            while (k < s.length() && s.charAt(j) == s.charAt(k)) {
                k++;
            }
            int c2 = k - j;
            count += Math.min(c1, c2);
            i = j;
        }
        return count;
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

    public void helper(String input) {
        if (input == null) return;
        String[] arr = input.split(" ");
        if (arr.length != 3) return;
        int n = Integer.parseInt(arr[0]), p = Integer.parseInt(arr[1]),
                q = Integer.parseInt(arr[2]);
        List<String> res = new ArrayList<>();
        for (int i = 1; i <= n; i++){
            boolean modePQ = i % p == 0 || i % q == 0;
            boolean containsPQ = containsPQ(i, p, q);
            if (modePQ && containsPQ) {
                res.add("OUTTHINK");
            } else if (modePQ) {
                res.add("OUT");
            } else if (containsPQ) {
                res.add("THINK");
            } else {
                res.add(String.valueOf(i));
            }
        }
        System.out.println(String.join(",", res));
    }

    private boolean containsPQ(int i, int p, int q) {
        while (i > 0) {
            if (i % 10 == p || i % 10 == q) {
                return true;
            }
            i /= 10;
        }
        return false;
    }

    void helper2() throws IOException {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        String s = in.readLine();
        if (s == null || s.isEmpty()) return;
        String[] strarr = s.split(",");
        int n = strarr.length;
        if (n < 3) return;
        String p1 = strarr[n - 2], p2 = strarr[n - 1];
        Map<String, String> relations = new HashMap<>();
        for (int i = 0; i < n - 2; i++) {
            String manager = strarr[i].split("\\->")[0];
            String person = strarr[i].split("\\->")[1];
            relations.put(person, manager);
        }
        Set<String> p1m = new HashSet<>();
        while (relations.get(p1) != null) {
            p1m.add(relations.get(p1));
            p1 = relations.get(p1);
        }
        while (relations.get(p2) != null) {
            if (p1m.contains(relations.get(p2))) {
                System.out.println(relations.get(p2));
                break;
            }
            p2 = relations.get(p2);
        }
    }

    class Person {
        String name;
        Person manager;
    }

    static int maxLength(int[] a, int k) {
        if (a == null || a.length == 0 || k == 0) return 0;
        int i = 0, j = 0, sum = 0, max = 0;
        while (j < a.length) {
            while (sum > k && i < j) {
                sum -= a[i];
                i++;
            }
            sum += a[j++];
            if (sum <= k) {
                max = Math.max(max, j - i);
            }
        }
        return max;
    }

    static int findMutationDistance(String start, String end, String[] bank) {
        if (start == null || end == null) return -1;
        if (start.equals(end)) return 0;
        char[] genes = new char[] {'A','C','T','G'};
        Queue<String> queue = new LinkedList<>();
        Set<String> set = new HashSet<>();
        Set<String> bankSet = new HashSet<>();
        queue.offer(start);
        queue.offer("");
        set.add(start);
        Collections.addAll(bankSet, bank);
        int dis = 0;
        while (!queue.isEmpty()) {
            String cur = queue.poll();
            if (cur.equals(end)) {
                return dis;
            } else if (cur.isEmpty()) {
                if (queue.isEmpty()) {
                    break;
                } else {
                    queue.offer("");
                    dis++;
                }
            } else {
                for (int i = 0; i < cur.length(); i++) {
                    char[] chars = cur.toCharArray();
                    for (char gene : genes) {
                        if (chars[i] != gene) {
                            chars[i] = gene;
                            String next = String.valueOf(chars);
                            if (!set.contains(next) && bankSet.contains(next)) {
                                queue.offer(next);
                                set.add(next);
                            }
                        }
                    }
                }
            }
        }
        return -1;
    }
}
