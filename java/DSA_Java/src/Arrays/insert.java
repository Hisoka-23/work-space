package Arrays;

import java.util.Arrays;

class Demo111{
	static int[] insertAtLast(int a[], int element) {
		
		int i, b[] = new int[a.length+1];
		for(i=0; i<a.length; i++) {
			b[i] = a[i];
		}
		b[i] = element;
		return b;
	}
	
	static int[] insertAtFirst(int a[], int element) {
		int i,b[] = new int[a.length+1];
		b[0] = element;
		for(i=0; i<a.length; i++) {
			b[i+1] = a[i];
		}
		return b;
	}
	
	static int[] insertAtLoction(int a[], int element, int location) {
		int i, k=0, b[] = new int[a.length+1];
		for(i=0; i<location; i++) {
			b[k++] = a[i];
		}
		b[k++] = element;
		for(i=location; i<a.length; i++) {
			b[k++]=a[i];
		}
		return b;
	}
}

public class insert {

	public static void main(String[] args) {
		int a[] = {10, 20, 30, 40, 50};
		System.out.println(Arrays.toString(a));
		//System.out.println(Arrays.toString(Demo111.insertAtLast(a, 60)));
		System.out.println(Arrays.toString(Demo111.insertAtFirst(a, 5)));
		System.out.println(Arrays.toString(Demo111.insertAtLoction(a, 999, 3)));
	}
	
}
