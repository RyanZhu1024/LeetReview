package leet;

/**
 * Created by RyanZhu on 2/17/16.
 */
public class MyHashTable<K, V> {
    private int capacity = 32;
    private int size = 1;

    private HashEntry<K, V>[] table;

    private static class HashEntry<K, V> {
        private K key;
        private V value;

        public HashEntry(K key, V value) {
            this.key = key;
            this.value = value;
        }

        public K getKey() {
            return key;
        }

        public V getValue() {
            return value;
        }
    }

    public MyHashTable() {
        this.table = new HashEntry[capacity];
        for (int i = 0; i < capacity; i++) {
            this.table[i] = null;
        }
    }

    public V get(K key) {
        int hash = hash(key);
        if (table[hash] == null) return null;
        return table[hash].getValue();
    }

    private int hash(K key) {
        int hash = key.hashCode() % capacity;
        while (table[hash] != null && !table[hash].getKey().equals(key)) {
            hash = (hash + 1) % capacity;
        }
        return hash;
    }

    private void rehash() {
        capacity *= 2;
        HashEntry<K, V>[] tempTable= this.table;
        this.table=new HashEntry[capacity];
        for (int i = 0; i < tempTable.length; i++) {
            HashEntry<K, V> item = tempTable[i];
            if (item != null) {
                int hash = hash(item.getKey());
                this.table[hash] = item;
            }
        }
    }

    public void put(K key, V value) {
        if (size == capacity) {
            rehash();
        }
        int hash = key.hashCode() % capacity;
        while (table[hash] != null && table[hash].getKey() != key) {
            hash = (hash + 1) % capacity;
        }
        HashEntry entry = new HashEntry(key, value);
        table[hash] = entry;
        size++;
    }

    public static void main(String[] args) {
        MyHashTable<String, Integer> t = new MyHashTable<>();
        for (int i = 0; i < 100; i++) {
            t.put(String.valueOf(i), i);
        }
        for (int i = 0; i < 100; i++) {
            System.out.println(t.get(String.valueOf(i)));
        }
    }
}
