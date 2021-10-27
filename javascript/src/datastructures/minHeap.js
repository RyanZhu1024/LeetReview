class MinHeap {
  constructor() {
    this.list = [];
  }
  heapify = (i) => {
    const left = 2 * i + 1;
    const right = 2 * i + 2;
    const length = this.list.length;
    let smallestIndex = i;
    if (left < length && this.list[left] < this.list[smallestIndex]) {
      smallestIndex = left;
    }
    if (right < length && this.list[right] < this.list[smallestIndex]) {
      smallestIndex = right;
    }
    if (smallestIndex !== i) {
      this._swap(smallestIndex, i);
      this.heapify(smallestIndex);
    }
  }

  _swap = (i, j) => {
    [this.list[i], this.list[j]] = [this.list[j], this.list[i]]
  }
  insertNode = (node) => {
    this.list.push(node);
    for (let i = Math.floor(this.list.length / 2 - 1); i >= 0; i--) {
      this.heapify(i);
    }
  }
  deleteNode = (node) => {
    const nodeIndex = this.list.findIndex(n => n === node);
    if (nodeIndex > -1) {
      this._swap(nodeIndex, this.list.length - 1);
      this.list.splice(this.list.length - 1);
      for (let i = Math.floor(this.list.length / 2 - 1); i >= 0; i--) {
        this.heapify(i);
      }
    }
  }
  getMin = () => {
    return this.list[0];
  }
  deleteMin = () => {
    this.deleteNode(this.list[0])
  }
  popMin = () => {
    const minNode = this.getMin();
    this.deleteMin();
    return minNode;
  }
  isEmpty = () => {
    return this.list.length === 0;
  }
  getList = () => {
    return this.list;
  }
  size = () => {
    return this.list.length;
  }
}