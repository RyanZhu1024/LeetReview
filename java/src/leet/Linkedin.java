package leet;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by RyanZhu on 7/27/16.
 */
public class Linkedin {
    class Point {
        int x;
        int y;

        Point() {
            x = 0;
            y = 0;
        }

        Point(int a, int b) {
            x = a;
            y = b;
        }
    }

    public int maxPoints(Point[] points) {
        // Write your code here
        if (points == null || points.length == 0) {
            return 0;
        } else {
            int max = 0;
            for (int i = 0; i < points.length; i++) {
                Map<Double, Integer> map = new HashMap<>();
                map.put(Double.MAX_VALUE, 0);
                int samepoints = 1;
                for (int j = i + 1; j < points.length; j++) {
                    if (points[i].x == points[j].x && points[i].y == points[j].y) {
                        samepoints++;
                    } else if (points[i].x == points[j].x) {
                        map.put(Double.MAX_VALUE, map.get(Double.MAX_VALUE) + 1);
                    } else {
                        double slope = (double) (points[i].y - points[j].y) / (double) (points[i].x - points[j].x);
                        if (map.containsKey(slope)) {
                            map.put(slope, map.get(slope) + 1);
                        } else {
                            map.put(slope, 1);
                        }
                    }
                }
                int localMax = 0;
                for (Double key : map.keySet()) {
                    localMax = Math.max(localMax, map.get(key));
                }
                max = Math.max(max, localMax + samepoints);
            }
            return max;
        }
    }
}
