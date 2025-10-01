package Arrays;

import java.util.Arrays;

class Demo333{
	static int[] updateElementAtLocation(int a[], int location,int element) {
		int b[] = new int[a.length];
		for(int i=0; i<a.length; i++) {
			b[i] = a[i];
		}
		if(location >=0 && location<a.length) {
			b[location]=element;
		}
		return b;
	}
	
	static int[] updateElement(int a[],int oldElement, int newElement) {
		int i,b[] = new int[a.length];
		for(i=0;i<a.length;i++) {
			b[i]=a[i];
		}
		for(i=0;i<b.length;i++) {
			if(b[i]==oldElement) {
				b[i] = newElement;
				break;
			}
		}
		return b;
	}
}

public class update {

	public static void main(String[] args) {
		int a[] = {10, 11, 12, 13, 15, 16};
		System.out.println(Arrays.toString(a));
//		System.out.println(Arrays.toString(Demo333.updateElementAtLocation(a,0,999)));
//		System.out.println(Arrays.toString(Demo333.updateElementAtLocation(a,1,999)));
//		System.out.println(Arrays.toString(Demo333.updateElementAtLocation(a,2,999)));
//		System.out.println(Arrays.toString(Demo333.updateElementAtLocation(a,3,999)));
		System.out.println(Arrays.toString(Demo333.updateElement(a, 12, 23)));
	}
	
}
