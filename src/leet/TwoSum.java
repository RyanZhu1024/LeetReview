package leet;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by RyanZhu on 2/27/16.
 */
public class TwoSum {
    Map<Integer,Integer> map=new HashMap<>();

    // Add the number to an internal data structure.
    public void add(int number) {
        if(map.get(number)==null) map.put(number,1);
        else map.put(number,map.get(number)+1);
    }

    // Find if there exists any pair of numbers which sum is equal to the value.
    public boolean find(int value) {
        for (Integer key : map.keySet()) {
            int num2=value-key;
            if(num2==key&&map.get(key)>1){
                return true;
            }
            if(num2!=key&&map.get(num2)!=null){
                return true;
            }
        }
        return false;
    }
}
