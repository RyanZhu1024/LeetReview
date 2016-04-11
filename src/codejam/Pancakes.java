package codejam;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Scanner;

/**
 * Created by RyanZhu on 4/9/16.
 */
public class Pancakes {

    private static int solve(String line) {
        int j = line.length() - 1, count = 0;
        while (j >= 0) {
            if (line.charAt(j) == '+') {
                j--;
            } else if (line.charAt(j) == '-' && line.charAt(0) == '-') {
                count++;
                line = flip(line, j);
                j--;
            } else if (line.charAt(j) == '-' && line.charAt(0) == '+') {
                int i = 0;
                while (i < j && line.charAt(i) == '+') {
                    i++;
                }
                line = flip(line, i - 1);
                line = flip(line, j);
                j -= i;
                count += 2;
            }
        }
        return count;
    }

    private static String flip(String line, int right) {
        StringBuilder sb = new StringBuilder();
        for (int i = right; i >= 0; i--) {
            sb.append(line.charAt(i) == '-' ? '+' : '-');
        }
        return sb.toString() + line.substring(right + 1);
    }

    public static void main(String[] args) {
//        solve("++++-");
        Scanner in = new Scanner(new BufferedReader(new InputStreamReader(System.in)));
        int t = Integer.valueOf(in.nextLine());  // Scanner has functions to read ints, longs, strings, chars, etc.
        for (int i = 1; i <= t; ++i) {
            String line = in.nextLine();
            int num = solve(line);
            System.out.println("Case #" + i + ": " + num);
        }
    }
}
