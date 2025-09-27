package Arrays;

import java.util.Scanner;

public class Liner {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		System.out.print("Enter the size of array : ");
		int size = obj.nextInt();
		
		System.out.print("Enter the "+size+" elements : ");
		int a[] = new int[size];
		for(int i=0; i<a.length; i++) {
			a[i] = obj.nextInt();
		}
		
		System.out.print("Ente the key value : ");
		int key = obj.nextInt();
		
		int index = -1;
		
		for(int i=0; i<a.length; i++) {
			if(a[i] == key) {
				index = i;
				break;
			}
		}
		
		System.out.println("the index : "+index);
		
	}
	
}
