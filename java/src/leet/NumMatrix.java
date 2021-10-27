package leet;

/**
 * Created by RyanZhu on 8/21/16.
 */
public class NumMatrix {

    int[][] tree;
    int[][] matrix;

    public NumMatrix(int[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) return;
        this.matrix = matrix;
        int m = matrix.length, n = matrix[0].length;
        this.tree = new int[m + 1][n + 1];
        for (int i = 1; i < m + 1; i++) {
            for (int j = 1; j < n + 1; j++) {
                updateTree(i, j, matrix[i - 1][j - 1]);
            }
        }
    }

    void updateTree(int row, int col, int val) {
        int m = tree.length, n = tree[0].length;
        for (int i = row; i < m; i = getNext(i)) {
            for (int j = col; j < n; j = getNext(j)) {
                tree[i][j] += val;
            }
        }
    }

    int getNext(int i) {
        return i + (i & -i);
    }

    int getParent(int i) {
        return i - (i & -i);
    }

    public void update(int row, int col, int val) {
        int diff = val - matrix[row][col];
        matrix[row][col] = val;
        updateTree(row + 1, col + 1, diff);
    }

    int preSum(int row, int col) {
        row++;
        col++;
        int sum = 0;
        for (int i = row; i > 0; i = getParent(i)) {
            for (int j = col; j > 0; j = getParent(j)) {
                sum += tree[i][j];
            }
        }
        return sum;
    }

    public int sumRegion(int row1, int col1, int row2, int col2) {
        return preSum(row2, col2) + preSum(row1 - 1, col1 - 1) - preSum(row1 - 1, col2) - preSum(row2, col1 - 1);
    }

    public void printTree() {
        for (int i = 1; i < tree.length; i++) {
            for (int j = 1; j < tree[0].length; j++) {
                System.out.print(tree[i][j] + ",");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        NumMatrix nm = new NumMatrix(new int[][]{{3,0,1,4,2},{5,6,3,2,1},{1,2,0,1,5},{4,1,0,1,7},{1,0,3,0,5}});
        nm.printTree();
        System.out.println(nm.sumRegion(2,1,4,3));
        nm.update(3,2,2);
        System.out.println(nm.sumRegion(2,1,4,3));
    }

}
