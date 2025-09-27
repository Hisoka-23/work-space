package Arrays;

import java.util.Scanner;

public class MaxAndMin {
	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		System.out.println("Enter the size of array...");
		int size = obj.nextInt();
		
		System.out.println("Enter "+size+" elements");
		int a[] = new int[size];
		for(int i=0; i<a.length; i++) {
			a[i] = obj.nextInt();
		}
		
		int max = a[0];
		int min = a[0];
		for(int i=0; i<a.length; i++) {
			if(max < a[i]) {
				max = a[i];
			}
			
			if(min > a[i]) {
				min = a[i];
			}
		}
		System.out.println("max : "+max);
		
		System.out.println("min : "+min);
		
	}	
}
