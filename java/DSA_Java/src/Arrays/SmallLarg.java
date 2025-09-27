package Arrays;

import java.util.Scanner;

public class SmallLarg {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		System.out.println("Enter Size of Array : ");
		
		int size = obj.nextInt();
		
		int a[] = new int[size];
		
		System.out.println("Enter "+size+" elements");
		
		for(int i=0; i<a.length; i++) {
			a[i] = obj.nextInt();
		}
		
		//sort
		int temp;
		for(int i=0; i<a.length; i++) {
			for(int j=i+1; j<a.length; j++) {
				if(a[i] > a[j]) {
					temp = a[i];
					a[i] = a[j];
					a[j] = temp;
				}
			}
		}
		
		//logic
		int low,high;
		low=0;
		high=a.length-1;
		while(low < high) {
			System.out.print(a[low]+" "+a[high]+" ");
			low++;
			high--;
		}
		
		System.out.println();
		for(int i=0; i<a.length; i++) {
			System.out.print(a[i]+" ");
		}
		
	}
	
}
