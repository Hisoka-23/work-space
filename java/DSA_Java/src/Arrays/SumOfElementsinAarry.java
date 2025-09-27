package Arrays;

import java.util.Scanner;

public class SumOfElementsinAarry {
	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		System.out.println("Enter size of the array :");
		int size = obj.nextInt();
		
		System.out.println("Enter "+size+" elements");
		int a[] = new int[size];
		for(int i=0; i<a.length; i++) {
			a[i] = obj.nextInt();
		}
		
		int sum = 0;
		for(int i=0; i<a.length; i++) {
			sum=sum+a[i];
		}
		System.out.println("Sum of elements : "+sum);
		
	}	
}
