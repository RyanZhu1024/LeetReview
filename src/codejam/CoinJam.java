package codejam;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by RyanZhu on 4/9/16.
 */
public class CoinJam {
    public static void main(String[] args) {
//        Scanner in = new Scanner(new BufferedReader(new InputStreamReader(System.in)));
//        int t = Integer.parseInt(in.nextLine());  // Scanner has functions to read ints, longs, strings, chars, etc.
//        for (int i = 1; i <= t; ++i) {
//            String line = in.nextLine();
//            //length
//            int N = Integer.parseInt(line.split(" ")[0]);
//            //count
//            int J = Integer.parseInt(line.split(" ")[1]);
            List<Result> last = solve(16, 15);
//            System.out.println("Case #" + i + ":");
            for (Result result : last) {
                System.out.println(result.binary + " " + String.join(" ", result.divisors));
            }
//        }
    }

    private static List<Result> solve(int N, int J) {
        int max = (int) (Math.pow(2, N) - 1);
        List<Result> results = new ArrayList<>();
        while (max > 0 && J > 0) {
            String binaryStr = Integer.toBinaryString(max);
            List<String> list = new LinkedList<>();
            for (int i = 2; i <= 10; i++) {
                if (isPrime(Long.parseLong(binaryStr, i), list)) {
                    break;
                }
                if (i == 10) {
                    Result result = new Result();
                    result.binary = binaryStr;
                    result.divisors = list;
                    results.add(result);
                    J--;
                }
            }
            max--;
        }
        return results;
    }

    private static boolean isPrime(long n, List<String> list) {
        if (n == 0 || n == 1) return false;
        for (long i = 2; i < n; i++) {
            if (n % i == 0) {
                list.add(String.valueOf(i));
                return false;
            }
        }
        return true;
    }

    private static class Result {
        String binary;
        List<String> divisors = new ArrayList<>();
    }
}
