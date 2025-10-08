package Rotations;

import java.util.Arrays;
import java.util.Scanner;

class Demo2{
	static void reverse(int a[], int s, int e) {
		int temp;
		while(s<e) {
			temp=a[s];
			a[s]=a[e];
			a[e]=temp;
			s++;
			e--;
		}
	}
	static int[] rotateLeft_reversal(int a[], int r) {
		r=r%a.length;
		reverse(a,0,r-1);
		reverse(a,r,a.length-1);
		reverse(a,0,a.length);
		return a;
	}
}

public class Rotations5 {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		int a[] = {1, 2, 3, 4, 5};
		
		System.out.println("Enter Number of rotation : ");
		int n = obj.nextInt();
		
		System.out.println("Before Rotation ==> "+Arrays.toString(a));
		a=Demo2.rotateLeft_reversal(a,n);
		System.out.println("After Rotation ==> "+Arrays.toString(a));
		
	}
	
}
