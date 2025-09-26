package Arrays;

import java.util.Scanner;

public class SumOfElementsinAarry {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		System.out.println("Enter size of the array");
		
		int size = obj.nextInt();
		
		int a[] = new int[size];
		
		System.out.println("Enter "+size+"elements...");
		
		for(int i=0; i<a.length; i++) {
			a[i] = obj.nextInt();
		}
		
		int sum = 0;
		
		for(int i=0; i<a.length; i++) {
			sum=sum+a[i];
		}
		
		System.out.println(sum);
		
	}
	
}
