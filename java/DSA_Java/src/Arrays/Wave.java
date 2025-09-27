package Arrays;

import java.util.Scanner;

public class Wave {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		System.out.println("Enter size of the array : ");
		
		int size = obj.nextInt();
		
		System.out.println("Enter "+size+" elements in array");
		
		int a[] = new int[size];
		for(int i=0; i<a.length; i++) {
			a[i] = obj.nextInt();
		}
		
		//sort
		int i,temp;
		for(i=0; i<a.length; i++) {
			for(int j=i+1; j<a.length; j++) {
				temp = a[i];
				a[i] = a[j];
				a[j] = temp;
			}
		}
		
		//logic
		int t;
		System.out.print(a[0]+" ");
		for( i=1; i<a.length-1; i=i+2) {
			t = a[i];
			a[i] = a[i+1];
			a[i+1]  = t;
			System.out.print(a[i]+" "+a[i+1]+" ");
		}
		System.out.print(a[i]);
		
	}
	
}
