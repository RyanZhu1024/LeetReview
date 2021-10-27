package codejam;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;

/**
 * Created by RyanZhu on 4/8/16.
 */
public class CountSheep {
    private static int solve(int n) {
        if (n == 0) {
            return -1;
        } else {
            int i = 0;
            int x;
            Set<Integer> digitsSet = new HashSet<>();
            while (digitsSet.size() != 10) {
                i++;
                x = i * n;
                while (x != 0) {
                    int digit = x % 10;
                    x /= 10;
                    digitsSet.add(digit);
                }
            }
            return i * n;
        }
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(new BufferedReader(new InputStreamReader(System.in)));
        int t = in.nextInt();  // Scanner has functions to read ints, longs, strings, chars, etc.
        for (int i = 1; i <= t; ++i) {
            int n = in.nextInt();
            int last = solve(n);
            System.out.println("Case #" + i + ": " + (last == -1 ? "INSOMNIA" : last));
        }
    }
}
