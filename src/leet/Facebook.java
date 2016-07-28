package leet;

import java.util.*;

/**
 * Created by rzhu on 7/5/16.
 */
public class Facebook {
    public static void main(String[] args) {
        Facebook facebook = new Facebook();
        System.out.println(facebook.multiplay(5,-123));
    }

    public int multiplay(int x1, int x2) {
        if (x1 == 0 || x2 == 0) {
            return 0;
        }
        int symbol = ((x1 > 0) ^ (x2 > 0)) ? -1 : 1;
        long lx1 = Math.abs((long) x1);
        long lx2 = Math.abs((long) x2);
        long ans = 0;
        while (lx2 > 1) {
            long power = 1, temp = lx1;
            while ((power << 1) < lx2) {
                power <<= 1;
                temp <<= 1;
            }
            lx2 -= power;
            ans += temp;
        }
        if (lx2 == 1) {
            ans += lx1;
        }
        if (symbol == 1) {
            return (int)ans;
        } else {
            return (int)-ans;
        }
    }
    public int divide(int dividend, int divisor) {
        if (divisor == 0) {
            return Integer.MAX_VALUE;
        }
        if (dividend == Integer.MIN_VALUE && divisor == -1) {
            return Integer.MAX_VALUE;
        }
        int symbol = ((dividend > 0 && divisor < 0) || (dividend < 0 && divisor > 0)) ? -1 : 1;
        long d1 = Math.abs((long) dividend);
        long d2 = Math.abs((long) divisor);
        int ans = 0;
        while (d1 >= d2) {
            long power = 1, temp = d2;
            while ((temp << 1) < d1) {
                power <<= 1;
                temp <<= 1;
            }
            d1 -= temp;
            ans += power;
        }
        return ans * symbol;
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

    private class ArrayKey {
        int[] keys;

        public ArrayKey(String s) {
            keys = new int[26];
            for (int i = 0; i < s.length(); i++) {
                int index = s.charAt(i) - 97;
                keys[index]++;
            }
        }

        @Override
        public boolean equals(Object k2) {
            return Arrays.equals(keys, ((ArrayKey) k2).keys);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(keys);
        }

    }

    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> result = new ArrayList<>();
        if (strs == null || strs.length == 0) {
            return result;
        } else {
            Map<ArrayKey, List<String>> map = new HashMap<>();
            for (int i = 0; i < strs.length; i++) {
                ArrayKey key = new ArrayKey(strs[i]);
                if (!map.containsKey(key)) {
                    map.put(key, new ArrayList<>());
                }
                map.get(key).add(strs[i]);
            }
            for (List<String> list : map.values()) {
                result.add(list);
            }
            return result;
        }
    }

    public int combine123to100() {
        int[] coins = new int[]{1, 2, 5};
        int n = 100;
        return combine123to100Helper(coins, n, 0);
    }

    public int combine123to100DP() {
        int[] coins = new int[]{1, 2, 5};
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
        for (int i = 0; i < 4; i++) dp[i][0] = 1;
        for (int i = 1; i <= 3; i++) {
            for (int j = 1; j <= 100; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j - coins[i - 1] >= 0) {
                    dp[i][j] += dp[i][j - coins[i - 1]];
                }
            }
        }
        return dp[3][100];
    }

    private int combine123to100Helper(int[] coins, int n, int begin) {
        if (n == 0) return 1;
        if (n < 0) return 0;
        else {
            int sum = 0;
            for (int i = begin; i < coins.length; i++) {
                sum += combine123to100Helper(coins, n - coins[i], i);
            }
            return sum;
        }
    }


    //Definition for an interval.
    public static class Interval {
        int start;
        int end;

        Interval() {
            start = 0;
            end = 0;
        }

        Interval(int s, int e) {
            start = s;
            end = e;
        }
    }


    public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        List<Interval> result = new ArrayList<>();
        if (intervals == null || intervals.isEmpty()) {
            result.add(newInterval);
            return result;
        }
        int start = findLastLessEqualThanStart(intervals, newInterval);
        int end = findFirstGreaterEqualThanEnd(intervals, newInterval);
        if (start == -1 && end == -1) {
            result.add(newInterval);
        } else if (start == -1) {
            if (intervals.get(end).start <= newInterval.end) {
                newInterval.end = intervals.get(end).end;
                end += 1;
            }
            result.add(newInterval);
            for (int i = end; i < intervals.size(); i++) {
                result.add(intervals.get(i));
            }
        } else if (end == -1) {
            if (intervals.get(start).end >= newInterval.start) {
                newInterval.start = intervals.get(start).start;
                start -= 1;
            }
            for (int i = 0; i <= start; i++) {
                result.add(intervals.get(i));
            }
            result.add(newInterval);
        } else {
            if (intervals.get(start).end >= newInterval.start) {
                newInterval.start = intervals.get(start).start;
                start -= 1;
            }
            if (intervals.get(end).start <= newInterval.end) {
                newInterval.end = intervals.get(end).end;
                end += 1;
            }
            for (int i = 0; i <= start; i++) {
                result.add(intervals.get(i));
            }
            result.add(newInterval);
            for (int i = end; i < intervals.size(); i++) {
                result.add(intervals.get(i));
            }
        }
        return result;
    }

    int findLastLessEqualThanStart(List<Interval> intervals, Interval newInterval) {
        int i = 0, j = intervals.size() - 1;
        while (i + 1 < j) {
            int mid = i + (j - i) / 2;
            int midStart = intervals.get(mid).start;
            if (newInterval.start > midStart) {
                i = mid;
            } else if (newInterval.start < midStart) {
                j = mid - 1;
            } else {
                return mid;
            }
        }
        if (intervals.get(j).start <= newInterval.start) {
            return j;
        } else if (intervals.get(i).start <= newInterval.start) {
            return i;
        } else {
            return -1;
        }
    }

    int findFirstGreaterEqualThanEnd(List<Interval> intervals, Interval newInterval) {
        int i = 0, j = intervals.size() - 1;
        while (i + 1 < j) {
            int mid = i + (j - i) / 2;
            int midEnd = intervals.get(mid).end;
            if (newInterval.end < midEnd) {
                j = mid;
            } else if (newInterval.end > midEnd) {
                i = mid + 1;
            } else {
                return mid;
            }
        }
        if (intervals.get(i).end >= newInterval.end) {
            return i;
        } else if (intervals.get(j).end >= newInterval.end) {
            return j;
        } else {
            return -1;
        }
    }

    /**
     * sort by start time and min heap by end time
     * time: nlogn
     * space: k the number of rooms required
     *
     * @param intervals
     * @return
     */
    public int minMeetingRooms(Interval[] intervals) {
        if (intervals == null || intervals.length == 0) {
            return 0;
        } else {
            Arrays.sort(intervals, new Comparator<Interval>() {
                @Override
                public int compare(Interval i1, Interval i2) {
                    return i1.start - i2.start;
                }
            });
            PriorityQueue<Interval> queue = new PriorityQueue<>(new Comparator<Interval>() {
                @Override
                public int compare(Interval i1, Interval i2) {
                    return i1.end - i2.end;
                }
            });
            for (Interval i : intervals) {
                if (!queue.isEmpty() && i.start >= queue.peek().end) {
                    queue.poll();
                }
                queue.offer(i);
            }
            return queue.size();
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

    class TrieNode {
        // Initialize your data structure here.
        TrieNode[] children;
        boolean isWord;
        public TrieNode() {
            children = new TrieNode[26];
            isWord = false;
        }
    }

    public class Trie {
        private TrieNode root;

        public Trie() {
            root = new TrieNode();
        }

        // Inserts a word into the trie.
        public void insert(String word) {
            TrieNode node = root;
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (node.children[c - 'a'] == null) {
                    node.children[c - 'a'] = new TrieNode();
                }
                node = node.children[c - 'a'];
            }
            node.isWord = true;
        }

        // Returns if the word is in the trie.
        public boolean search(String word) {
            TrieNode node = findNode(word);
            return node == null ? false : node.isWord;
        }

        // Returns if there is any word in the trie
        // that starts with the given prefix.
        public boolean startsWith(String prefix) {
            TrieNode node = findNode(prefix);
            return node == null ? false : true;
        }

        TrieNode findNode(String str) {
            TrieNode node = root;
            for (int i = 0; i < str.length(); i++) {
                char c = str.charAt(i);
                if (node.children[c - 'a'] == null) {
                    return null;
                }
                node = node.children[c - 'a'];
            }
            return node;
        }
    }

    public class WordDictionary {

        class TrieNode {
            TrieNode[] children;
            boolean isWord;
            public TrieNode() {
                children = new TrieNode[26];
                isWord = false;
            }
        }

        TrieNode root = new TrieNode();

        // Adds a word into the data structure.
        public void addWord(String word) {
            TrieNode node = root;
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (node.children[c - 'a'] == null) {
                    node.children[c - 'a'] = new TrieNode();
                }
                node = node.children[c - 'a'];
            }
            node.isWord = true;
        }

        // Returns if the word is in the data structure. A word could
        // contain the dot character '.' to represent any one letter.
        public boolean search(String word) {
            Queue<TrieNode> queue = new LinkedList<>();
            TrieNode dummy = new TrieNode();
            queue.offer(root);
            queue.offer(dummy);
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (c == '.') {
                    while (!queue.isEmpty()) {
                        TrieNode node = queue.poll();
                        if (node == dummy) break;
                        for (TrieNode n : node.children) {
                            if (n != null) {
                                queue.offer(n);
                            }
                        }
                    }
                } else {
                    while (!queue.isEmpty()) {
                        TrieNode node = queue.poll();
                        if (node == dummy) break;
                        if (node.children[c - 'a'] != null) {
                            queue.offer(node.children[c - 'a']);
                        }
                    }
                }
                queue.offer(dummy);
            }
            while (!queue.isEmpty()) {
                if (queue.poll().isWord) {
                    return true;
                }
            }
            return false;
        }
    }

}
