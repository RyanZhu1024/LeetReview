package leet;

import java.util.TreeSet;

import static leet.Facebook.TreeNode;

/**
 * Created by RyanZhu on 25/09/2016.
 */
public class SundaySession {
    public static void main(String[] args) {
        System.out.println(maxModularSubarray(new int[]{3,3,9,9,5}, 7));
    }


    public int countNodes(TreeNode node) {
        if (node == null) return 0;
        int lh = getLeftHight(node.left);
        int rh = getRightHeight(node.right);

        if (lh == rh) {
            return (2 << lh) - 1;
        } else {
            return countNodes(node.left) + countNodes(node.right) + 1;
        }
    }

    public int getLeftHight(TreeNode node) {
        int h = 0;
        while (node != null) {
            node = node.left;
            h++;
        }
        return h;
    }

    public int getRightHeight(TreeNode node) {
        int h = 0;
        while (node != null) {
            node = node.right;
            h++;
        }
        return h;
    }
    private static int maxModularSubarray(int[] arr, int M) {
        int[] prefix = new int[arr.length];
        prefix[0] = arr[0];
        for (int i = 1; i < arr.length; i++) {
            prefix[i] = prefix[i - 1] + arr[i];
        }
        for (int i = 0; i < prefix.length; i++) {
            prefix[i] = prefix[i] % M;
        }
        TreeSet<Integer> tree = new TreeSet<>();
        tree.add(prefix[0]);
        int max = 0;
        for (int i = 1; i < prefix.length; i++) {
            int cur = prefix[i];
            Integer pre = tree.higher(cur);
            if (pre != null) {
                max = Math.max(max, (cur - pre + M) % M);
            } else {
                max = Math.max(max, prefix[i]);
            }
        }
        return max;
    }
}
