package Arrays;

import java.util.Scanner;

public class ReplaceElement {
	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		System.out.println("Enter size of array");
		int size = obj.nextInt();
		
		System.out.println("Enter "+size+" elemensts");
		int a[] = new int[size];
		for(int i=0; i<a.length; i++) {
			a[i] = obj.nextInt();
		}
		
		System.out.println("these are old elements...");
		for(int i=0; i<a.length; i++) {
			System.out.print(a[i]+" ");
		}
		
		System.out.println();
		
		int oldE, newE;
		System.out.println("Enter old elements");
		oldE = obj.nextInt();
		
		System.out.println("Enter new elements");
		newE = obj.nextInt();
		
		for(int i=0; i<a.length; i++) {
			if(oldE == a[i]) {
				a[i] = newE;
				break;
			}
		}
		
		System.out.println("these are new elements...");
		for(int i=0; i<a.length; i++) {
			System.out.print(a[i]+" ");
		}
		
	}	
}
