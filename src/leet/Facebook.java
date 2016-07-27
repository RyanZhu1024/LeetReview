package leet;

import java.util.*;

/**
 * Created by rzhu on 7/5/16.
 */
public class Facebook {
    public static void main(String[] args) {
        Facebook facebook = new Facebook();
        List<Interval> intervals = new ArrayList<>();
        intervals.add(new Interval(2,4));
        intervals.add(new Interval(5,7));
        intervals.add(new Interval(8,10));
        intervals.add(new Interval(11,13));
        System.out.println(facebook.insert(intervals,new Interval(3,6)));
    }

    private class ArrayKey {
        int[] keys;
        public ArrayKey (String s) {
            keys = new int[26];
            for (int i = 0; i < s.length(); i++) {
                int index = s.charAt(i) - 97;
                keys[index]++;
            }
        }

        @Override
        public boolean equals(Object k2) {
            return Arrays.equals(keys, ((ArrayKey)k2).keys);
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
            for(List<String> list : map.values()) {
                result.add(list);
            }
            return result;
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


     //Definition for an interval.
     public static class Interval {
         int start;
         int end;
         Interval() { start = 0; end = 0; }
         Interval(int s, int e) { start = s; end = e; }
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
     * @param intervals
     * @return
     */
    public int minMeetingRooms(Interval[] intervals) {
        if (intervals == null || intervals.length == 0) {
            return 0;
        } else {
            Arrays.sort(intervals, new Comparator<Interval>(){
                @Override
                public int compare(Interval i1, Interval i2) {
                    return i1.start - i2.start;
                }
            });
            PriorityQueue<Interval> queue = new PriorityQueue<>(new Comparator<Interval>(){
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
}
