package Rotations;

import java.util.Arrays;
import java.util.Scanner;

class Demo1{
	static int[] rotate_left(int a[], int r) {
		r=r%a.length;
		int i, n=a.length;
		
		int temp[] = new int[n];
		for(i=0; i<n; i++) {
			temp[i] = a[(i+r)%n];
		}
		
		for(i=0;i<n; i++) {
			a[i] = temp[i];
		}
		
		return a;
	}
	
	static int[] rotate_rigth(int a[], int r) {
		r=r%a.length;
		int i, n=a.length;
		
		int temp[] = new int[n];
		for(i=0; i<n; i++) {
			temp[(i+r)%n] = a[i];
		}
		
		for(i=0;i<n; i++) {
			a[i] = temp[i];
		}
		
		return a;
	}
}

public class Rotations4 {
	
	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		int a[] = {1, 2, 3, 4, 5};
		
		System.out.println("Enter Number of rotation : ");
		int n = obj.nextInt();
		
		System.out.println("Before Rotation ==> "+Arrays.toString(a));
		a=Demo1.rotate_rigth(a,n);
		System.out.println("After Rotation ==> "+Arrays.toString(a));
		
	}

}
