package Arrays;

import java.util.Arrays;

class Demo1{
	static boolean equals(int a[], int b[]) {
		for(int i=0; i<a.length; i++) {
			if(a[i]!=b[i]) {
				return false;
			}
		}
		return true;
	}
}

public class Arraysbasic {

	public static void main(String[] args) {
		System.out.println(Demo1.equals(new int[] {1,2,3}, new int[] {1,2,3}));
		System.out.println(Demo1.equals(new int[] {1,2,3}, new int[] {4,5,6}));
		System.out.println(Demo1.equals(new int[] {1,2,3}, new int[] {1,5,3}));
		System.out.println(Demo1.equals(new int[] {1,2,3}, new int[] {1,2,9}));
		int a[] = {1,2,3};
		int b[] = {3,2,1};
		System.out.println(Demo1.equals(a, b));//false
		Arrays.sort(a);
		Arrays.sort(b);
		System.out.println(Demo1.equals(a, b));//true
		
		System.out.println(Arrays.equals(a, b));
	}
	
}
