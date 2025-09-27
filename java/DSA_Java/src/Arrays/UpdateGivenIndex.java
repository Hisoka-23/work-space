package Arrays;

import java.util.Scanner;

public class UpdateGivenIndex {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		System.out.print("Enter the size of array : ");
		int size = obj.nextInt();
		
		int i,index,newValue;
		
		System.out.println("Enter "+size+" the elements");
		int a[] = new int[size];
		for(i=0; i<a.length; i++){
			a[i] = obj.nextInt();
		}
		
		System.out.println("array before updating ");
		for(i=0; i<a.length; i++) {
			System.out.print(a[i]+" ");
		}
		
		System.out.println();
		
		System.out.println("Enter the index valued : ");
		index = obj.nextInt();
		if(index >=0 && index < a.length) {
			System.out.println("Enter the new value : ");
			newValue = obj.nextInt();
			a[index] = newValue;
		}
		
		System.out.println("array after updating ");
		for(i=0; i<a.length; i++) {
			System.out.print(a[i]+" ");
		}
		
	}
	
}
