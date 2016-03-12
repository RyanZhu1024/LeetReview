package leet;

import java.util.LinkedList;

/**
 * Created by RyanZhu on 3/9/16.
 */
public class Mock {
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    public int rob(TreeNode root) {
        if (root == null) return 0;
        int prev = 0, cur = 0, max = 0, nic = 0, nin = 0;
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        nic++;
        while (!queue.isEmpty()) {
            if (nic > 0) {
                TreeNode node = queue.pollFirst();
                cur += node.val;
                max = Math.max(prev, cur);
                if (node.left != null) {
                    queue.add(node.left);
                    nin++;
                }
                if (node.right != null) {
                    queue.add(node.right);
                    nin++;
                }
                nic--;
            } else {
                int temp = cur;
                cur = prev;
                prev = temp;
                nic = nin;
                nin = 0;
            }
        }
        return max;
    }

    public int numDecodings(String s) {
        if (s == null || s.isEmpty() || s.charAt(0) == '0') return 0;
        if (s.length() == 1) return 1;
        int num = 1, lastNum = -1, lastNumCount = 0;
        for (int i = s.length()-1; i >= 0; i--) {
            int curNum=Character.getNumericValue(s.charAt(i));
            if(curNum==0){
                if(i>0){
                    int beforeNum=Character.getNumericValue(s.charAt(i-1));
                    if(beforeNum>2||beforeNum==0){
                        return 0;
                    }else{
                        lastNum=-1;
                        i--;
                    }
                }else{
                    return 0;
                }
            }else{
                if(lastNum==-1){
                    lastNum=curNum;
                    lastNumCount=1;
                }else{
                    int combine=curNum*10+lastNum;
                    int temp=num;
                    if(combine<=26){
                        num+=lastNumCount;
                    }
                    lastNumCount=temp;
                    lastNum=curNum;
                }
            }
        }
        return num;
    }


    public static void main(String[] args) {
        System.out.println(new Mock().numDecodings("110"));
    }
}
