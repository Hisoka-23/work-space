package Arrays;

import java.util.Arrays;
import java.util.Scanner;

public class SortV1 {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		System.out.println("Enter array size:");
		
		int size = obj.nextInt();
		
		int a[] = new int[size];
		
		System.out.println("Enter "+size+" elements in array");
		
		for(int i=0; i<a.length; i++) {
			a[i] = obj.nextInt();
		}
		
		System.out.print("Array Elements Before Sorting: ");
		for(int i=0; i<a.length; i++) {
			System.out.print(a[i]+" ");
		}
		
		//version 1 => sort the data in asc order
		int temp1;
		for(int i=0; i<a.length; i++) {
			for(int j=i+1; j<a.length; j++) {
				if(a[i] > a[j]) {
					temp1 = a[i];
					a[i] = a[j];
					a[j] = temp1;
				}
			}
		}
		
		System.out.println();
		System.out.print("Array Elements After Sorting version 1: ");
		for(int i=0; i<a.length; i++) {
			System.out.print(a[i]+" ");
		}
		
		//version 2 => sort the data in dsc order
		int temp2;
		for(int i=0; i<a.length; i++) {
			for(int j=i+1; j<a.length; j++) {
				if(a[i] < a[j]) {
					temp2 = a[i];
					a[i] = a[j];
					a[j] = temp2;
				}
			}
		}
		System.out.println();
		System.out.print("Array Elements After Sorting version 2: ");
		for(int i=0; i<a.length; i++) {
			System.out.print(a[i]+" ");
		}
		
		//version 3 => sort the data in asc order
		Arrays.sort(a);
		System.out.println();
		System.out.print("Array Elements Ater Sorting version 3: ");
		for(int i=0; i<a.length; i++) {
			System.out.print(a[i]+" ");
		}
		
		//version 4 => sort the data in asc order
		Arrays.sort(a);
		System.out.println();
		System.out.print("Array Elements Ater Sorting version 3: ");
		for(int i=a.length-1; i>=0; i--) {
			System.out.print(a[i]+" ");
		}
		
		//version 5 => sort the data in asc order
		Arrays.sort(a,a.length/2, a.length);
		System.out.println();
		System.out.print("Array Elements Ater Sorting version 3: ");
		for(int i=0; i<a.length; i++) {
			System.out.print(a[i]+" ");
		}
	}
	
}
